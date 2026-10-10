package io.teknek.deliverance.generator2;

import com.google.common.base.Preconditions;
import io.dropwizard.metrics5.MetricRegistry;
import io.dropwizard.metrics5.Timer;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.kv.CacheExecutionMode;
import io.teknek.deliverance.tensor.kv.AttentionPattern;
import io.teknek.deliverance.tensor.kv.KvCacheSession;
import io.teknek.deliverance.tensor.kv.KvReadView;
import io.teknek.deliverance.tensor.kv.KvWriteCursor;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.BatchDotProduct;
import io.teknek.deliverance.tensor2.TensorRef;
import io.teknek.deliverance.safetensors.LoraLayerDelta;
import io.teknek.deliverance.generator.ForwardPhase;

import java.util.List;
import java.util.Collections;
import java.util.Optional;
import java.util.concurrent.ForkJoinTask;
import java.util.function.Consumer;

/** TensorRef-native base for self-attention backed by KV cache v2. */
public abstract class KvCacheSelfAttention2 extends BaseCausalSelfAttention2 {
    private static final int DEFAULT_GROUPED_DECODE_QKV_SPLIT_SIZE = 8;
    protected final AbstractModel model;
    protected final int layerIndex;
    protected final io.teknek.deliverance.safetensors.Config config;
    protected final TensorRef queryAttentionWeights;
    protected final TensorRef keyAttentionWeights;
    protected final TensorRef valueAttentionWeights;
    protected final TensorRef outputProjectionWeights;
    protected final MetricRegistry metricRegistry;
    protected final String queryWeightName;
    protected final String keyWeightName;
    protected final String valueWeightName;
    protected final String outputWeightName;
    protected final int attentionLength;
    protected final int kvLength;
    protected final int numberOfHeads;
    protected final int numberOfKeyValueHeads;
    protected final int headGroupSize;
    protected final float attentionScale;
    protected final Lighter lighter;
    private final PackedBlockAttention2 packedBlockAttention;

    protected KvCacheSelfAttention2(AbstractModel model, int layerIndex, TensorRef queryAttentionWeights,
            TensorRef keyAttentionWeights, TensorRef valueAttentionWeights, TensorRef outputProjectionWeights,
            Lighter lighter, MetricRegistry metricRegistry, String queryWeightName, String keyWeightName,
            String valueWeightName, String outputWeightName) {
        super(lighter);
        this.model = java.util.Objects.requireNonNull(model, "model");
        this.layerIndex = layerIndex;
        this.config = model.getConfig();
        this.queryAttentionWeights = java.util.Objects.requireNonNull(queryAttentionWeights, "queryAttentionWeights");
        this.keyAttentionWeights = java.util.Objects.requireNonNull(keyAttentionWeights, "keyAttentionWeights");
        this.valueAttentionWeights = java.util.Objects.requireNonNull(valueAttentionWeights, "valueAttentionWeights");
        this.outputProjectionWeights = java.util.Objects.requireNonNull(outputProjectionWeights,
                "outputProjectionWeights");
        this.lighter = java.util.Objects.requireNonNull(lighter, "lighter");
        this.metricRegistry = java.util.Objects.requireNonNull(metricRegistry, "metricRegistry");
        this.queryWeightName = queryWeightName;
        this.keyWeightName = keyWeightName;
        this.valueWeightName = valueWeightName;
        this.outputWeightName = outputWeightName;
        this.attentionLength = model.getLocalAttentionLength();
        this.kvLength = model.getLocalKvLength();
        this.numberOfHeads = model.getLocalNumberOfHeads();
        this.numberOfKeyValueHeads = model.getLocalNumberOfKeyValueHeads();
        Preconditions.checkArgument(numberOfKeyValueHeads > 0 && numberOfHeads % numberOfKeyValueHeads == 0,
                "Attention heads must be divisible by KV heads");
        this.headGroupSize = numberOfHeads / numberOfKeyValueHeads;
        this.attentionScale = config.attentionMultiplier != null
                ? config.attentionMultiplier
                : (float) (1.0 / StrictMath.sqrt(config.headSize));
        this.packedBlockAttention = new PackedBlockAttention2(model, metricRegistry, layerIndex);
    }

    @Override
    public TensorRef forward(TensorRef input, int startPosition, KvCacheSession kvSession,
            Optional<Consumer<List<TensorRef>>> tensorReducer, ForwardPhase phase) {
        Preconditions.checkArgument(input.dims() == 2 && input.shape().last() == config.embeddingLength,
                "Attention input must be [batch, embeddingLength]");
        Preconditions.checkArgument(startPosition >= 0 && startPosition <= kvSession.length(),
                "startPosition must be within KV session length");
        CacheExecutionMode mode = phase == ForwardPhase.DECODE
                ? CacheExecutionMode.DECODE_UPDATE_CACHE
                : CacheExecutionMode.PREFILL_UPDATE_CACHE;
        int batchSize = input.shape().first();
        Preconditions.checkArgument(mode != CacheExecutionMode.DECODE_UPDATE_CACHE || batchSize == 1,
                "decode update expects one token at a time");

        try (Timer.Context ignored = InferenceProfiler.timer(metricRegistry,
                "kvcacheselfattention2.forward").time()) {
            model.emitLayerDebug(layerIndex, "attention_projection_input", input);
            TensorRef[] qkv = projectQkv(input, phase);
            try {
                normalizeQueryKey(qkv[0], qkv[1]);
                model.emitLayerDebug(layerIndex, "query_normalized", qkv[0]);
                model.emitLayerDebug(layerIndex, "key_normalized", qkv[1]);
                applyRotaryEmbedding(qkv[0], qkv[1], startPosition);
                model.emitLayerDebug(layerIndex, "query_rope", qkv[0]);
                model.emitLayerDebug(layerIndex, "key_rope", qkv[1]);
                if (writesCache(mode)) {
                    writeKvRows(kvSession, mode, qkv[1], qkv[2], startPosition);
                }
                TensorRef attended = attend(qkv[0], qkv[1], qkv[2], kvSession, startPosition, mode,
                        tensorReducer, phase);
                try {
                    model.emitLayerDebug(layerIndex, "attention_value", attended);
                    return projectOutput(attended, tensorReducer, phase);
                } finally {
                    attended.close();
                }
            } finally {
                closeAll(qkv);
            }
        }
    }

    protected void normalizeQueryKey(TensorRef query, TensorRef key) {
    }

    protected void applyRotaryEmbedding(TensorRef query, TensorRef key, int startPosition) {
    }

    protected final TensorRef attend(TensorRef query, TensorRef currentKeys, TensorRef currentValues,
            KvCacheSession kvSession, int startPosition, CacheExecutionMode mode,
            Optional<Consumer<List<TensorRef>>> tensorReducer, ForwardPhase phase) {
        int batchSize = query.shape().first();
        TensorRef attended = model.makeDenseTensorRef(batchSize, attentionLength);
        try {
            if (startPosition == 0) {
                packedBlockAttention.forward(attended, query, currentKeys, currentValues, 0, batchSize, numberOfHeads,
                        numberOfKeyValueHeads, config.headSize, attentionScale, config.attnLogitSoftCapping,
                        mode != CacheExecutionMode.DENOISE_BLOCK_NO_UPDATE);
                return attended;
            }
            try (KvReadView readView = kvSession.readView(layerIndex, startPosition, AttentionPattern.CAUSAL)) {
                if (mode == CacheExecutionMode.DECODE_UPDATE_CACHE && batchSize == 1) {
                    TensorRef[] prefixKeyPages = readView.keyPageRefs();
                    TensorRef[] prefixValuePages = null;
                    try {
                        prefixValuePages = readView.valuePageRefs();
                    compositeOps.decodePagedAttention(attended, query,
                            appendCurrentPage(prefixKeyPages, currentKeys),
                            appendCurrentPage(prefixValuePages, currentValues), startPosition + 1,
                            numberOfHeads, numberOfKeyValueHeads, config.headSize, attentionScale,
                            config.attnLogitSoftCapping, model.getPool().getUnderlying(),
                            model.getPool().getCoreCount());
                    } finally {
                        closePrefixPages(prefixKeyPages);
                        closePrefixPages(prefixValuePages);
                    }
                    return attended;
                }
                int capacity = startPosition + batchSize;
                try (TensorRef packedKeys = lighter.allocate(readView.keyDType(), TensorShape.of(capacity, kvLength));
                     TensorRef packedValues = lighter.allocate(readView.valueDType(), TensorShape.of(capacity, kvLength))) {
                    try (Timer.Context ignoredPack = InferenceProfiler.timer(metricRegistry,
                            "kvcacheselfattention2.pack_kv").time()) {
                        copyPrefixRows(readView, packedKeys, packedValues, startPosition);
                        copyCurrentRows(currentKeys, packedKeys, startPosition);
                        copyCurrentRows(currentValues, packedValues, startPosition);
                    }
                    packedBlockAttention.forward(attended, query, packedKeys, packedValues, startPosition, batchSize,
                            numberOfHeads, numberOfKeyValueHeads, config.headSize, attentionScale,
                            config.attnLogitSoftCapping, mode != CacheExecutionMode.DENOISE_BLOCK_NO_UPDATE);
                }
                return attended;
            }
        } catch (RuntimeException | Error e) {
            attended.close();
            throw e;
        }
    }

    private void copyPrefixRows(KvReadView readView, TensorRef packedKeys, TensorRef packedValues, int rowCount) {
        for (int row = 0; row < rowCount; row++) {
            try (TensorRef keyRow = readView.keyRowRef(row); TensorRef valueRow = readView.valueRowRef(row)) {
                copyRow(keyRow, 0, packedKeys, row, kvLength);
                copyRow(valueRow, 0, packedValues, row, kvLength);
            }
        }
    }

    private TensorRef[] appendCurrentPage(TensorRef[] prefixPages, TensorRef currentPage) {
        TensorRef[] pages = new TensorRef[prefixPages.length + 1];
        System.arraycopy(prefixPages, 0, pages, 0, prefixPages.length);
        pages[prefixPages.length] = currentPage;
        return pages;
    }

    private void closePrefixPages(TensorRef[] pages) {
        if (pages == null) {
            return;
        }
        for (TensorRef page : pages) {
            page.close();
        }
    }

    private void copyCurrentRows(TensorRef source, TensorRef destination, int destinationRowStart) {
        if (source.dType() == destination.dType()) {
            for (int row = 0; row < source.shape().first(); row++) {
                copyRow(source, row, destination, destinationRowStart + row, source.shape().last());
            }
            return;
        }
        try (TensorRef converted = lighter.reshape(source, destination.dType())) {
            for (int row = 0; row < converted.shape().first(); row++) {
                copyRow(converted, row, destination, destinationRowStart + row, converted.shape().last());
            }
        }
    }

    protected final TensorRef[] projectQkv(TensorRef input, ForwardPhase phase) {
        Preconditions.checkArgument(input.dims() == 2 && input.shape().last() == config.embeddingLength,
                "Attention input must be [batch, embeddingLength]");
        int batchSize = input.shape().first();
        TensorRef query = model.makeDenseTensorRef(batchSize, attentionLength);
        TensorRef key = model.makeDenseTensorRef(batchSize, kvLength);
        TensorRef value = model.makeDenseTensorRef(batchSize, kvLength);
        try (Timer.Context ignored = InferenceProfiler.timer(metricRegistry,
                "kvcacheselfattention2.qkv_projection").time()) {
            if (config.isGQA) {
                projectGqaQkvPrefill(input, query, key, value, model.primaryTensorOperations().parallelSplitSize(), phase);
            } else {
                project(query, input, queryAttentionWeights, config.embeddingLength, attentionLength,
                        "kvcacheselfattention2.q_projection", model.primaryTensorOperations().parallelSplitSize(), phase);
                project(key, input, keyAttentionWeights, config.embeddingLength, kvLength,
                        "kvcacheselfattention2.k_projection", model.primaryTensorOperations().parallelSplitSize(), phase);
                project(value, input, valueAttentionWeights, config.embeddingLength, kvLength,
                        "kvcacheselfattention2.v_projection", model.primaryTensorOperations().parallelSplitSize(), phase);
            }
            applyLora(queryWeightName, query, input, phase,
                    "kvcacheselfattention2.q_lora");
            applyLora(keyWeightName, key, input, phase,
                    "kvcacheselfattention2.k_lora");
            applyLora(valueWeightName, value, input, phase,
                    "kvcacheselfattention2.v_lora");
            model.emitLayerDebug(layerIndex, "query_projection", query);
            model.emitLayerDebug(layerIndex, "key_projection", key);
            model.emitLayerDebug(layerIndex, "value_projection", value);
            return new TensorRef[] {query, key, value};
        } catch (RuntimeException | Error e) {
            query.close();
            key.close();
            value.close();
            throw e;
        }
    }

    private void projectGqaQkvPrefill(TensorRef input, TensorRef query, TensorRef key, TensorRef value,
            int splitSize, ForwardPhase phase) {
        ForkJoinTask<?> queryTask = model.getPool().getUnderlying().submit(() -> project(query, input,
                queryAttentionWeights, config.embeddingLength, attentionLength,
                "kvcacheselfattention2.q_projection", splitSize, phase));
        ForkJoinTask<?> keyTask = model.getPool().getUnderlying().submit(() -> project(key, input,
                keyAttentionWeights, config.embeddingLength, kvLength,
                "kvcacheselfattention2.k_projection", splitSize, phase));
        ForkJoinTask<?> valueTask = model.getPool().getUnderlying().submit(() -> project(value, input,
                valueAttentionWeights, config.embeddingLength, kvLength,
                "kvcacheselfattention2.v_projection", splitSize, phase));
        queryTask.join();
        keyTask.join();
        valueTask.join();
    }

    private void projectGqaQkvGroupedDecode(TensorRef input, TensorRef query, TensorRef key, TensorRef value) {
        model.runChunks("kvcacheselfattention2.qkv_projection.grouped_decode", 0, attentionLength,
                groupedDecodeQkvSplitSize(), Optional.empty(), (chunkStart, chunkSize) -> {
            try (Timer.Context ignoredQ = InferenceProfiler.timer(metricRegistry,
                    "kvcacheselfattention2.q_projection").time()) {
                lighter.dotProductRows(query, input, queryAttentionWeights, 0, config.embeddingLength,
                        chunkStart, chunkSize, chunkStart);
            }
            int kvChunkSize = Math.min(chunkSize, kvLength - chunkStart);
            if (kvChunkSize <= 0) {
                return;
            }
            try (Timer.Context ignoredK = InferenceProfiler.timer(metricRegistry,
                    "kvcacheselfattention2.k_projection").time()) {
                lighter.dotProductRows(key, input, keyAttentionWeights, 0, config.embeddingLength,
                        chunkStart, kvChunkSize, chunkStart);
            }
            try (Timer.Context ignoredV = InferenceProfiler.timer(metricRegistry,
                    "kvcacheselfattention2.v_projection").time()) {
                lighter.dotProductRows(value, input, valueAttentionWeights, 0, config.embeddingLength,
                        chunkStart, kvChunkSize, chunkStart);
            }
        });
    }

    private void project(TensorRef output, TensorRef input, TensorRef weight, int inputLength,
            int outputLength, String metricName, int splitSize, ForwardPhase phase) {
        model.runChunks(metricName, 0, outputLength, splitSize, Optional.empty(),
                (chunkStart, chunkSize) -> {
            try (Timer.Context ignored = InferenceProfiler.timer(metricRegistry, metricName).time()) {
                if (phase == ForwardPhase.PREFILL && input.shape().first() > 1) {
                    lighter.batchDotProduct(new BatchDotProduct()
                            .result(output).a(input).b(weight)
                            .aColumnOffset(0).bColumnOffset(0).columnLength(inputLength)
                            .resultRowOffset(0).bRowOffset(chunkStart).rowChunkSize(chunkSize));
                } else {
                    lighter.dotProductRows(output, input, weight, 0, inputLength, chunkStart, chunkSize,
                            chunkStart);
                }
            }
        });
    }

    private int groupedDecodeQkvSplitSize() {
        int configured = model.groupedDecodeQkvSplitSize().orElse(DEFAULT_GROUPED_DECODE_QKV_SPLIT_SIZE);
        int poolSize = model.getPool() == null ? 1 : model.getPool().getCoreCount();
        return Math.max(1, Math.min(Math.min(configured, poolSize), attentionLength));
    }

    protected final TensorRef projectOutput(TensorRef attended, Optional<Consumer<List<TensorRef>>> tensorReducer,
            ForwardPhase phase) {
        Preconditions.checkArgument(attended.dims() == 2 && attended.shape().last() == attentionLength,
                "Attention output must be [batch, attentionLength]");
        TensorRef output = model.makeDenseTensorRef(attended.shape().first(), config.embeddingLength);
        output.memorySegment().fill((byte) 0);
        TensorRef projectionInput = attended;
        boolean closeProjectionInput = false;
        try {
            if (attended.dType() != model.getWorkingQType()) {
                InferenceProfiler.counter(metricRegistry,
                        "causalselfattention.maybe_quantize.output_projection.copy_or_quantize").inc();
                projectionInput = lighter.reshape(attended, model.getWorkingQType());
                closeProjectionInput = true;
            } else {
                InferenceProfiler.counter(metricRegistry,
                        "causalselfattention.maybe_quantize.output_projection.read_only").inc();
            }
            model.emitLayerDebug(layerIndex, "attention_output_projection_input", projectionInput);
            try (Timer.Context ignoredOutput = InferenceProfiler.timer(metricRegistry,
                    "kvcacheselfattention2.output_projection").time()) {
                TensorRef projectionInputForChunks = projectionInput;
                model.runChunks("kvcacheselfattention2.output_projection", 0, config.embeddingLength,
                        model.primaryTensorOperations().parallelSplitSize(), Optional.empty(), (chunkStart, chunkSize) ->
                        {
                             projectChunk(output, projectionInputForChunks, outputProjectionWeights,
                                     attentionLength, chunkStart, chunkSize, phase);
                        });
            }
            applyLora(outputWeightName, output, projectionInput, phase,
                    "kvcacheselfattention2.o_lora");
            if (model.getTensorParallelContext().enabled()) {
                throw new UnsupportedOperationException("TensorRef attention output projection all-reduce is not ported");
            }
            model.emitLayerDebug(layerIndex, "attention_output", output);
            tensorReducer.ifPresent(func -> func.accept(Collections.singletonList(output)));
            return output;
        } catch (RuntimeException | Error e) {
            output.close();
            throw e;
        } finally {
            if (closeProjectionInput) {
                projectionInput.close();
            }
        }
    }

    private void projectChunk(TensorRef output, TensorRef input, TensorRef weight, int inputLength,
            int chunkStart, int chunkSize, ForwardPhase phase) {
        if (phase == ForwardPhase.PREFILL && input.shape().first() > 1) {
            lighter.batchDotProduct(new BatchDotProduct()
                    .result(output).a(input).b(weight)
                    .aColumnOffset(0).bColumnOffset(0).columnLength(inputLength)
                    .resultRowOffset(0).bRowOffset(chunkStart).rowChunkSize(chunkSize));
        } else {
            lighter.dotProductRows(output, input, weight, 0, inputLength, chunkStart, chunkSize, chunkStart);
        }
    }

    private void applyLora(String weightName, TensorRef output, TensorRef input,
            ForwardPhase phase, String metricName) {
        if (weightName != null) {
            model.activeLoraDeltaFor(weightName)
                    .ifPresent(value -> LoraDeltaApplier2.apply(model, lighter, output, input, value, phase, metricName));
        }
    }

    protected boolean writesCache(CacheExecutionMode mode) {
        return mode == CacheExecutionMode.PREFILL_UPDATE_CACHE
                || mode == CacheExecutionMode.DECODE_UPDATE_CACHE
                || mode == CacheExecutionMode.VERIFY_AND_UPDATE_CACHE;
    }

    private void writeKvRows(KvCacheSession kvSession, CacheExecutionMode mode, TensorRef keys, TensorRef values,
            int startPosition) {
        try (KvWriteCursor writer = kvSession.writer(mode)) {
            for (int row = 0; row < keys.shape().first(); row++) {
                try (TensorRef key = keys.slice(row); TensorRef value = values.slice(row)) {
                    writer.write(layerIndex, startPosition + row, key, value);
                }
            }
        }
    }
}
