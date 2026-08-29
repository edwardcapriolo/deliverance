package io.teknek.deliverance.generator;

import com.google.common.base.Preconditions;
import io.dropwizard.metrics5.MetricRegistry;
import io.dropwizard.metrics5.Timer;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.kv.AttentionPattern;
import io.teknek.deliverance.tensor.kv.CacheExecutionMode;
import io.teknek.deliverance.tensor.kv.KvCacheSession;
import io.teknek.deliverance.tensor.kv.KvReadView;
import io.teknek.deliverance.tensor.kv.KvWriteCursor;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.TensorOperations;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.InferenceProfiler;

import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.ForkJoinTask;
import java.util.function.Consumer;

/**
 * Self-attention implementation backed by KV cache v2 sessions.
 *
 * <p>This class intentionally mirrors the projection/provider path from {@link CausalSelfAttention} while keeping v2
 * cache mutation modes explicit. The implementation supports causal AR prefill/decode plus no-update bidirectional
 * active-block attention for diffusion denoising. It currently packs v2 read views into dense tensors for attention
 * scoring; page-backed v2 attention can replace that internals without changing callers.</p>
 */
public class KvCacheSelfAttention {
    private final AbstractModel model;
    private final int layerIndex;
    private final io.teknek.deliverance.safetensors.Config config;
    private final AbstractTensor queryAttnWeights;
    private final AbstractTensor keyAttnWeights;
    private final AbstractTensor valueAttnWeights;
    private final AbstractTensor outputProjectionWeights;
    private final ConfigurableTensorProvider configurableTensorProvider;
    private final MetricRegistry metricRegistry;
    private final String queryWeightName;
    private final String keyWeightName;
    private final String valueWeightName;
    private final String outputWeightName;
    private final int attentionLength;
    private final int kvLength;
    private final int numberOfHeads;
    private final int numberOfKeyValueHeads;
    private final int headGroupSize;
    private final float attentionScale;

    public KvCacheSelfAttention(AbstractModel model, int layerIndex, AbstractTensor queryAttnWeights,
            AbstractTensor keyAttnWeights, AbstractTensor valueAttnWeights, AbstractTensor outputProjectionWeights,
            ConfigurableTensorProvider configurableTensorProvider, MetricRegistry metricRegistry,
            String queryWeightName, String keyWeightName, String valueWeightName, String outputWeightName) {
        this.model = model;
        this.layerIndex = layerIndex;
        this.config = model.getConfig();
        this.queryAttnWeights = queryAttnWeights;
        this.keyAttnWeights = keyAttnWeights;
        this.valueAttnWeights = valueAttnWeights;
        this.outputProjectionWeights = outputProjectionWeights;
        this.configurableTensorProvider = configurableTensorProvider;
        this.metricRegistry = metricRegistry;
        this.queryWeightName = queryWeightName;
        this.keyWeightName = keyWeightName;
        this.valueWeightName = valueWeightName;
        this.outputWeightName = outputWeightName;
        this.attentionLength = model.getLocalAttentionLength();
        this.kvLength = model.getLocalKvLength();
        this.numberOfHeads = model.getLocalNumberOfHeads();
        this.numberOfKeyValueHeads = model.getLocalNumberOfKeyValueHeads();
        this.headGroupSize = numberOfHeads / numberOfKeyValueHeads;
        this.attentionScale = config.attentionMultiplier != null
                ? config.attentionMultiplier
                : (float) (1.0 / StrictMath.sqrt(config.headSize));

        configurableTensorProvider.get().registerModelTensor(queryAttnWeights);
        configurableTensorProvider.get().registerModelTensor(keyAttnWeights);
        configurableTensorProvider.get().registerModelTensor(valueAttnWeights);
        configurableTensorProvider.get().registerModelTensor(outputProjectionWeights);
    }

    public AbstractTensor forward(AbstractTensor input, int startPosition, KvCacheSession kvSession,
            Optional<Consumer<List<AbstractTensor>>> tensorReducer) {
        CacheExecutionMode mode = input.shape().first() == 1
                ? CacheExecutionMode.DECODE_UPDATE_CACHE
                : CacheExecutionMode.PREFILL_UPDATE_CACHE;
        ForwardPhase phase = mode == CacheExecutionMode.DECODE_UPDATE_CACHE ? ForwardPhase.DECODE : ForwardPhase.PREFILL;
        return forward(input, startPosition, kvSession, mode, tensorReducer, phase);
    }

    public AbstractTensor forward(AbstractTensor input, int startPosition, KvCacheSession kvSession,
            CacheExecutionMode mode, Optional<Consumer<List<AbstractTensor>>> tensorReducer, ForwardPhase phase) {
        Preconditions.checkArgument(mode == CacheExecutionMode.PREFILL_UPDATE_CACHE
                        || mode == CacheExecutionMode.DECODE_UPDATE_CACHE
                        || mode == CacheExecutionMode.DENOISE_BLOCK_NO_UPDATE
                        || mode == CacheExecutionMode.READ_PREFIX_NO_UPDATE
                        || mode == CacheExecutionMode.VERIFY_AND_UPDATE_CACHE,
                "KvCacheSelfAttention supports prefill/decode updates and denoise no-update, got %s", mode);
        Preconditions.checkArgument(input.dims() == 2 && input.shape().last() == config.embeddingLength);
        Preconditions.checkArgument(startPosition >= 0 && startPosition <= kvSession.length(),
                "startPosition must be within KV session length");
        int batchSize = input.shape().first();
        Preconditions.checkArgument(mode != CacheExecutionMode.DECODE_UPDATE_CACHE || batchSize == 1,
                "decode update expects one token at a time");

        try (Timer.Context ignored = InferenceProfiler.timer(metricRegistry, "kvcacheselfattention.forward").time();
             AbstractTensor projectionInput = model.maybeQuantizeReadOnly(input,
                     "transformerblock.maybe_quantize.pre_attention");
             AbstractTensor queryBatch = model.makeDenseTensor(batchSize, attentionLength);
             AbstractTensor keyBatch = model.makeDenseTensor(batchSize, kvLength);
             AbstractTensor valueBatch = model.makeDenseTensor(batchSize, kvLength);
             AbstractTensor attended = model.makeDenseTensor(batchSize, attentionLength)) {
            projectQkv(projectionInput, queryBatch, keyBatch, valueBatch, phase);
            applyRotaryEmbedding(queryBatch, keyBatch, startPosition);
            if (writesCache(mode)) {
                writeKvRows(kvSession, mode, keyBatch, valueBatch, startPosition);
            }
            attend(attended, queryBatch, keyBatch, valueBatch, kvSession, startPosition, batchSize, mode);
            return outputProjection(attended, tensorReducer, phase);
        }
    }

    private boolean writesCache(CacheExecutionMode mode) {
        return mode == CacheExecutionMode.PREFILL_UPDATE_CACHE || mode == CacheExecutionMode.DECODE_UPDATE_CACHE
                || mode == CacheExecutionMode.VERIFY_AND_UPDATE_CACHE;
    }

    private void projectQkv(AbstractTensor input, AbstractTensor queryBatch, AbstractTensor keyBatch,
            AbstractTensor valueBatch, ForwardPhase phase) {
        int splitSize = configurableTensorProvider.get().parallelSplitSize();
        if (config.isGQA) {
            try (Timer.Context ignoredQkv = InferenceProfiler.timer(metricRegistry,
                    "kvcacheselfattention.qkv_projection").time()) {
                ForkJoinTask<?> queryTask = model.getPool().getUnderlying().submit(() -> project(queryBatch, input,
                        queryAttnWeights, config.embeddingLength, attentionLength,
                        "kvcacheselfattention.q_projection", phase, splitSize));
                ForkJoinTask<?> keyTask = model.getPool().getUnderlying().submit(() -> project(keyBatch, input,
                        keyAttnWeights, config.embeddingLength, kvLength,
                        "kvcacheselfattention.k_projection", phase, splitSize));
                ForkJoinTask<?> valueTask = model.getPool().getUnderlying().submit(() -> project(valueBatch, input,
                        valueAttnWeights, config.embeddingLength, kvLength,
                        "kvcacheselfattention.v_projection", phase, splitSize));
                queryTask.join();
                keyTask.join();
                valueTask.join();
            }
        } else {
            AbstractTensor[] results = new AbstractTensor[] { queryBatch, keyBatch, valueBatch };
            AbstractTensor[] weights = new AbstractTensor[] { queryAttnWeights, keyAttnWeights, valueAttnWeights };
            TensorOperations projectionOps = projectionOperations(input, queryAttnWeights, phase);
            try (Timer.Context ignoredQkv = InferenceProfiler.timer(metricRegistry,
                    "kvcacheselfattention.qkv_projection").time()) {
                model.runChunks("kvcacheselfattention.qkv_projection", 0, attentionLength,
                        projectionOps.parallelSplitSize(), Optional.of(input), (chunkStart, chunkSize) ->
                                projectionOps.dotProductBatchChunk(results, input, weights, 0,
                                        config.embeddingLength, chunkStart, chunkSize));
            }
        }
        model.activeLoraDeltaFor(queryWeightName).ifPresent(delta -> LoraDeltaApplier.apply(model, queryBatch, input, delta));
        model.activeLoraDeltaFor(keyWeightName).ifPresent(delta -> LoraDeltaApplier.apply(model, keyBatch, input, delta));
        model.activeLoraDeltaFor(valueWeightName).ifPresent(delta -> LoraDeltaApplier.apply(model, valueBatch, input, delta));
        model.emitLayerDebug(layerIndex, "query_projection", queryBatch);
        model.emitLayerDebug(layerIndex, "key_projection", keyBatch);
        model.emitLayerDebug(layerIndex, "value_projection", valueBatch);
    }

    private void project(AbstractTensor output, AbstractTensor input, AbstractTensor weight, int inputLength,
            int outputLength, String metricName, ForwardPhase phase, int fallbackSplitSize) {
        TensorOperations projectionOps = projectionOperations(input, weight, phase);
        int splitSize = projectionOps == null ? fallbackSplitSize : projectionOps.parallelSplitSize();
        TensorOperations ops = projectionOps == null ? configurableTensorProvider.get() : projectionOps;
        try (Timer.Context ignored = InferenceProfiler.timer(metricRegistry, metricName).time()) {
            model.runChunks(metricName, 0, outputLength, splitSize, Optional.of(input), (chunkStart, chunkSize) ->
                    ops.dotProductChunk(output, input, weight, 0, inputLength, chunkStart, chunkSize));
        }
    }

    private TensorOperations projectionOperations(AbstractTensor input, AbstractTensor weight, ForwardPhase phase) {
        TensorOperations operations = model.prefillProjectionOperations(input, weight, phase);
        return operations == null ? configurableTensorProvider.get() : operations;
    }

    private void applyRotaryEmbedding(AbstractTensor queryBatch, AbstractTensor keyBatch, int startPosition) {
        try (Timer.Context ignored = InferenceProfiler.timer(metricRegistry, "kvcacheselfattention.rope").time()) {
            boolean handledByModel = model.applyRotaryEmbedding(queryBatch, keyBatch, startPosition, numberOfHeads,
                    numberOfKeyValueHeads, config.headSize, configurableTensorProvider.get());
            Preconditions.checkState(handledByModel || config.ropeFreqs.isPresent(),
                    "model must provide RoPE or config.ropeFreqs");
            if (!handledByModel) {
                applyConfiguredRope(queryBatch, keyBatch, startPosition);
            }
        }
    }

    private void applyConfiguredRope(AbstractTensor queryBatch, AbstractTensor keyBatch, int startPosition) {
        float[][] ropeFreqs = config.ropeFreqs.orElseThrow();
        int headPiece = config.headSize / 2;
        for (int row = 0; row < queryBatch.shape().first(); row++) {
            int positionOffset = (startPosition + row) * headPiece;
            try (AbstractTensor query = queryBatch.slice(row);
                 AbstractTensor key = keyBatch.slice(row)) {
                CausalSelfAttention.rotateRopeHeads(query, numberOfHeads, config.headSize, headPiece,
                        positionOffset, ropeFreqs);
                CausalSelfAttention.rotateRopeHeads(key, numberOfKeyValueHeads, config.headSize, headPiece,
                        positionOffset, ropeFreqs);
            }
        }
    }

    private void writeKvRows(KvCacheSession kvSession, CacheExecutionMode mode, AbstractTensor keyBatch,
            AbstractTensor valueBatch, int startPosition) {
        try (Timer.Context ignored = InferenceProfiler.timer(metricRegistry, "kvcacheselfattention.kv_cache_write").time();
             KvWriteCursor writer = kvSession.writer(mode)) {
            for (int row = 0; row < keyBatch.shape().first(); row++) {
                try (AbstractTensor keyRow = keyBatch.slice(row);
                     AbstractTensor valueRow = valueBatch.slice(row)) {
                    writer.write(layerIndex, startPosition + row, keyRow, valueRow);
                }
            }
        }
    }

    private void attend(AbstractTensor output, AbstractTensor queryBatch, AbstractTensor keyBatch,
            AbstractTensor valueBatch, KvCacheSession kvSession, int startPosition, int batchSize,
            CacheExecutionMode mode) {
        if (startPosition == 0) {
            fullSequenceAttention(output, queryBatch, keyBatch, valueBatch,
                    mode != CacheExecutionMode.DENOISE_BLOCK_NO_UPDATE);
            return;
        }
        try (KvReadView readView = kvSession.readView(layerIndex, startPosition, AttentionPattern.CAUSAL);
             AbstractTensor previousKeys = readView.copyVisibleKeys();
             AbstractTensor previousValues = readView.copyVisibleValues();
             AbstractTensor packedKeys = model.makeDenseTensor(TensorShape.of(startPosition + batchSize, kvLength));
             AbstractTensor packedValues = model.makeDenseTensor(TensorShape.of(startPosition + batchSize, kvLength))) {
            if (startPosition > 0) {
                packedKeys.copyFrom(previousKeys, 0, 0, (int) previousKeys.size());
                packedValues.copyFrom(previousValues, 0, 0, (int) previousValues.size());
            }
            packedKeys.copyFrom(keyBatch, 0, packedKeys.getOffset(startPosition, 0), (int) keyBatch.size());
            packedValues.copyFrom(valueBatch, 0, packedValues.getOffset(startPosition, 0), (int) valueBatch.size());
            if (mode == CacheExecutionMode.DENOISE_BLOCK_NO_UPDATE) {
                bidirectionalBlockAttentionWithPrefix(output, queryBatch, packedKeys, packedValues, startPosition,
                        batchSize);
            } else {
                causalAttentionWithPrefix(output, queryBatch, packedKeys, packedValues, startPosition, batchSize);
            }
        }
    }

    private void fullSequenceAttention(AbstractTensor output, AbstractTensor query, AbstractTensor key,
            AbstractTensor value, boolean causal) {
        try (Timer.Context ignored = InferenceProfiler.timer(metricRegistry,
                "kvcacheselfattention.attention").time()) {
            TensorOperations ops = configurableTensorProvider.get();
            int sequenceLength = (int) query.shape().first();
            output.clear();
            try (AbstractTensor scores = model.makeDenseTensor(1, sequenceLength)) {
                for (int row = 0; row < sequenceLength; row++) {
                    try (AbstractTensor queryRow = query.slice(row);
                         AbstractTensor outputRow = output.slice(row)) {
                        int visible = causal ? row + 1 : sequenceLength;
                        attentionRow(ops, outputRow, queryRow, key, value, visible);
                    }
                }
            }
        }
    }

    private void causalAttentionWithPrefix(AbstractTensor output, AbstractTensor query, AbstractTensor keys,
            AbstractTensor values, int startPosition, int batchSize) {
        try (Timer.Context ignored = InferenceProfiler.timer(metricRegistry,
                "kvcacheselfattention.attention").time()) {
            TensorOperations ops = configurableTensorProvider.get();
            output.clear();
            for (int row = 0; row < batchSize; row++) {
                try (AbstractTensor queryRow = query.slice(row);
                     AbstractTensor outputRow = output.slice(row)) {
                    attentionRow(ops, outputRow, queryRow, keys, values, startPosition + row + 1);
                }
            }
        }
    }

    private void bidirectionalBlockAttentionWithPrefix(AbstractTensor output, AbstractTensor query, AbstractTensor keys,
            AbstractTensor values, int startPosition, int batchSize) {
        try (Timer.Context ignored = InferenceProfiler.timer(metricRegistry,
                "kvcacheselfattention.attention").time()) {
            TensorOperations ops = configurableTensorProvider.get();
            output.clear();
            int visibleRows = startPosition + batchSize;
            for (int row = 0; row < batchSize; row++) {
                try (AbstractTensor queryRow = query.slice(row);
                     AbstractTensor outputRow = output.slice(row)) {
                    attentionRow(ops, outputRow, queryRow, keys, values, visibleRows);
                }
            }
        }
    }

    private void attentionRow(TensorOperations ops, AbstractTensor outputRow, AbstractTensor queryRow,
            AbstractTensor keys, AbstractTensor values, int visibleRows) {
        try (AbstractTensor scores = model.makeDenseTensor(1, visibleRows)) {
            for (int head = 0; head < numberOfHeads; head++) {
                int kvHead = Math.floorDiv(head, headGroupSize);
                int queryOffset = head * config.headSize;
                int kvOffset = kvHead * config.headSize;
                for (int keyPosition = 0; keyPosition < visibleRows; keyPosition++) {
                    try (AbstractTensor keyRow = keys.slice(keyPosition)) {
                        scores.set(ops.dotProduct(queryRow, keyRow, queryOffset, kvOffset, config.headSize), 0,
                                keyPosition);
                    }
                }
                ops.scaledSoftMax(scores, 0, visibleRows, attentionScale, config.attnLogitSoftCapping);
                ops.saxpy(scores, values, outputRow, kvOffset, queryOffset, config.headSize, 0, 0, visibleRows);
            }
        }
    }

    private AbstractTensor outputProjection(AbstractTensor attended,
            Optional<Consumer<List<AbstractTensor>>> tensorReducer, ForwardPhase phase) {
        AbstractTensor result = model.makeDenseTensor((int) attended.shape().first(), config.embeddingLength);
        try {
            try (AbstractTensor projectionInput = model.maybeQuantizeReadOnly(attended,
                    "causalselfattention.maybe_quantize.output_projection")) {
                TensorOperations outputOps = projectionOperations(projectionInput, outputProjectionWeights, phase);
                try (Timer.Context ignoredOutput = InferenceProfiler.timer(metricRegistry,
                        "kvcacheselfattention.output_projection").time()) {
                    model.runChunks("kvcacheselfattention.output_projection", 0, config.embeddingLength,
                            outputOps.parallelSplitSize(), Optional.of(projectionInput), (chunkStart, chunkSize) ->
                                    outputOps.dotProductChunk(result, projectionInput, outputProjectionWeights, 0,
                                            attentionLength, chunkStart, chunkSize));
                }
                model.activeLoraDeltaFor(outputWeightName).ifPresent(
                        delta -> LoraDeltaApplier.apply(model, result, projectionInput, delta));
            }
            AbstractTensor reduced = model.getTensorParallelContext().enabled()
                    ? model.getTensorParallelCollectives().allReduceSum("layer." + layerIndex + ".self_attn.o_proj", result)
                    : result;
            model.emitLayerDebug(layerIndex, "attention_output", reduced);
            tensorReducer.ifPresent(func -> func.accept(Collections.singletonList(reduced)));
            if (reduced != result) {
                result.close();
            }
            return reduced;
        } catch (RuntimeException | Error e) {
            result.close();
            throw e;
        }
    }
}
