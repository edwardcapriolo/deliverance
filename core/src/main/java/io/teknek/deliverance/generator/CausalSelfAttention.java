package io.teknek.deliverance.generator;

import com.codahale.metrics.MetricRegistry;
import com.codahale.metrics.Timer;
import com.google.common.base.Preconditions;
import io.teknek.deliverance.math.VectorMath;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.DecodeAttentionMode;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.model.TensorProviderKind;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.KvBufferCache;
import io.teknek.deliverance.tensor.KvPageTable;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.TensorOperations;
import io.teknek.deliverance.tensorlib.TensorPlan;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import net.jafama.FastMath;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.util.*;
import java.util.concurrent.ForkJoinTask;
import java.util.function.Consumer;

import static io.teknek.deliverance.tensor.DebugSupport.debug;

public class CausalSelfAttention extends BaseCausalSelfAttention {
    private static final Logger logger = LoggerFactory.getLogger(CausalSelfAttention.class);
    private static final String PACKED_PREFILL_PROPERTY = "deliverance.attention.packed-prefill";
    private static final VectorSpecies<Float> F32_SPECIES = FloatVector.SPECIES_PREFERRED;

    private final AbstractModel m;
    private final Config config;
    private final int layerIndex;
    private final Optional<AbstractTensor> queryAttnBias;
    private final Optional<AbstractTensor> keyAttnBias;

    private final Optional<AbstractTensor> valueAttnBias;
    private final Optional<AbstractTensor> outputProjectionBias;

    final AbstractTensor queryAttnWeights;
    final AbstractTensor keyAttnWeights;

    final AbstractTensor valueAttnWeights;

    private final AbstractTensor outputProjectionWeights;

    private final float attentionScale;
    private final int attentionLength;
    private final int kvLength;
    private final int numberOfHeads;
    private final int numberOfKeyValueHeads;
    private final int headGroupSize;

    private final AbstractTensor[] qkvResults;
    private final AbstractTensor[] qkvWeights;
    private final ConfigurableTensorProvider configurableTensorProvider;

    private final MetricRegistry metricRegistry;

    /**
     * Base tensor names for this layer's q/k/v/o projections (e.g.
     * {@code "model.layers.3.self_attn.q_proj.weight"}), used to look up an active LoRA adapter's
     * delta for this layer via {@link AbstractModel#activeLoraDeltaFor(String)}. {@code null} for
     * model families that haven't opted into LoRA runtime hot-swap (see step 4 plan Section 6) --
     * safe because {@code activeLoraDeltaFor} never consults the name when no adapter is active,
     * and non-opted-in families can never have one.
     */
    private final String queryWeightName;
    private final String keyWeightName;
    private final String valueWeightName;
    private final String outputWeightName;

    public CausalSelfAttention(
            AbstractModel m,
            int layerIndex,
            AbstractTensor queryAttnWeights,
            AbstractTensor keyAttnWeights,
            AbstractTensor valueAttnWeights,
            AbstractTensor outputProjectionWeights,
            ConfigurableTensorProvider configurableTensorProvider,
            MetricRegistry metricRegistry
    ) {
        this(
                m,
                layerIndex,
                queryAttnWeights,
                keyAttnWeights,
                valueAttnWeights,
                outputProjectionWeights,
                configurableTensorProvider,
                metricRegistry,
                null,
                null,
                null,
                null
        );
    }

    /** Variant carrying base tensor names for LoRA runtime hot-swap -- see step 4 plan Section 4.1. */
    public CausalSelfAttention(
            AbstractModel m,
            int layerIndex,
            AbstractTensor queryAttnWeights,
            AbstractTensor keyAttnWeights,
            AbstractTensor valueAttnWeights,
            AbstractTensor outputProjectionWeights,
            ConfigurableTensorProvider configurableTensorProvider,
            MetricRegistry metricRegistry,
            String queryWeightName,
            String keyWeightName,
            String valueWeightName,
            String outputWeightName
    ) {
        this(
                m,
                layerIndex,
                Optional.empty(),
                Optional.empty(),
                Optional.empty(),
                queryAttnWeights,
                keyAttnWeights,
                valueAttnWeights,
                Optional.empty(),
                outputProjectionWeights,
                configurableTensorProvider,
                metricRegistry,
                queryWeightName,
                keyWeightName,
                valueWeightName,
                outputWeightName
        );
    }

    public CausalSelfAttention(
            AbstractModel m,
            int layerIndex,
            Optional<AbstractTensor> queryAttnBias,
            Optional<AbstractTensor> keyAttnBias,
            Optional<AbstractTensor> valueAttnBias,
            AbstractTensor queryAttnWeights,
            AbstractTensor keyAttnWeights,
            AbstractTensor valueAttnWeights,
            Optional<AbstractTensor> outputProjectionBias,
            AbstractTensor outputProjectionWeights,
            ConfigurableTensorProvider configurableTensorProvider,
            MetricRegistry metricRegistry
    ) {
        this(
                m,
                layerIndex,
                queryAttnBias,
                keyAttnBias,
                valueAttnBias,
                queryAttnWeights,
                keyAttnWeights,
                valueAttnWeights,
                outputProjectionBias,
                outputProjectionWeights,
                configurableTensorProvider,
                metricRegistry,
                null,
                null,
                null,
                null
        );
    }

    /** Canonical constructor, carrying base tensor names for LoRA runtime hot-swap -- see step 4 plan Section 4.1. */
    public CausalSelfAttention(
            AbstractModel m,
            int layerIndex,
            Optional<AbstractTensor> queryAttnBias,
            Optional<AbstractTensor> keyAttnBias,
            Optional<AbstractTensor> valueAttnBias,
            AbstractTensor queryAttnWeights,
            AbstractTensor keyAttnWeights,
            AbstractTensor valueAttnWeights,
            Optional<AbstractTensor> outputProjectionBias,
            AbstractTensor outputProjectionWeights,
            ConfigurableTensorProvider configurableTensorProvider,
            MetricRegistry metricRegistry,
            String queryWeightName,
            String keyWeightName,
            String valueWeightName,
            String outputWeightName
    ) {
        this.m = m;
        this.layerIndex = layerIndex;
        this.config = m.getConfig();
        this.queryAttnBias = queryAttnBias;
        this.keyAttnBias = keyAttnBias;
        this.valueAttnBias = valueAttnBias;
        this.queryAttnWeights = queryAttnWeights;
        this.keyAttnWeights = keyAttnWeights;
        this.valueAttnWeights = valueAttnWeights;

        this.outputProjectionBias = outputProjectionBias;
        this.outputProjectionWeights = outputProjectionWeights;
        this.attentionLength = m.getLocalAttentionLength();
        this.kvLength = m.getLocalKvLength();
        this.numberOfHeads = m.getLocalNumberOfHeads();
        this.numberOfKeyValueHeads = m.getLocalNumberOfKeyValueHeads();
        this.headGroupSize = numberOfHeads / numberOfKeyValueHeads;

        this.attentionScale = config.attentionMultiplier != null ? config.attentionMultiplier : (float) (1.0 / StrictMath.sqrt(config.headSize));

        this.qkvResults = new AbstractTensor[3];
        this.qkvWeights = new AbstractTensor[] { queryAttnWeights, keyAttnWeights, valueAttnWeights };
        this.configurableTensorProvider = configurableTensorProvider;

        configurableTensorProvider.get().registerModelTensor(queryAttnWeights);
        configurableTensorProvider.get().registerModelTensor(keyAttnWeights);
        configurableTensorProvider.get().registerModelTensor(valueAttnWeights);
        configurableTensorProvider.get().registerModelTensor(outputProjectionWeights);

        this.metricRegistry = metricRegistry;
        this.queryWeightName = queryWeightName;
        this.keyWeightName = keyWeightName;
        this.valueWeightName = valueWeightName;
        this.outputWeightName = outputWeightName;
    }

    public AbstractTensor forward(AbstractTensor input, int startPosition, KvBufferCache.KvBuffer kvMem,
            Optional<Consumer<List<AbstractTensor>>> tensorReducer) {
        return forward(input, startPosition, kvMem, tensorReducer, ForwardPhase.DECODE);
    }

    public AbstractTensor forward(AbstractTensor input, int startPosition, KvBufferCache.KvBuffer kvMem,
            Optional<Consumer<List<AbstractTensor>>> tensorReducer, ForwardPhase phase) {
        Timer forwardTimer = InferenceProfiler.timer(metricRegistry, "causalselfattention.forward");
        try (Timer.Context ignored = forwardTimer.time()) {
        Preconditions.checkArgument(input.dims() == 2 && input.shape().last() == config.embeddingLength);
        int batchSize = input.shape().first();
        int splitSize = configurableTensorProvider.get().parallelSplitSize();
        try (AbstractTensor queryBatch = m.makeDenseTensor(batchSize, attentionLength);
                AbstractTensor tmpKeyBatch = m.makeDenseTensor(batchSize, kvLength);
                AbstractTensor tmpValBatch = m.makeDenseTensor(batchSize, kvLength);
                AbstractTensor valueBatch = m.makeDenseTensor(batchSize, attentionLength)) {
            if (config.isGQA) {
                Timer tm = metricRegistry.timer("causualselfattention.forward_gqa_querybatch_1");
                try (Timer.Context ignoredQkv = InferenceProfiler.timer(metricRegistry, "causalselfattention.qkv_projection").time()) {
                    ForkJoinTask<?> queryTask = m.getPool().getUnderlying().submit(() ->
                            VectorMath.pchunkMetrics(0, attentionLength, (chunkStart, chunkLength) ->
                                            configurableTensorProvider.get().dotProductChunk(queryBatch, input,
                                                    queryAttnWeights, 0, config.embeddingLength, chunkStart,
                                                    chunkLength),
                                    splitSize, tm, m.getPool()));
                    ForkJoinTask<?> keyTask = m.getPool().getUnderlying().submit(() ->
                            VectorMath.pchunk(0, kvLength, (chunkStart, chunkLength) -> {
                                Timer t = metricRegistry.timer("causualselfattention.forward_gqa_key_2");
                                try (Timer.Context context = t.time()) {
                                    configurableTensorProvider.get().dotProductChunk(tmpKeyBatch, input,
                                            keyAttnWeights, 0, config.embeddingLength, chunkStart, chunkLength);
                                }
                            }, splitSize, m.getPool()));
                    ForkJoinTask<?> valueTask = m.getPool().getUnderlying().submit(() ->
                            VectorMath.pchunk(0, kvLength, (chunkStart, chunkLength) -> {
                                Timer r = metricRegistry.timer("causualselfattention.forward_gqa_val_3");
                                try (Timer.Context context = r.time()) {
                                    configurableTensorProvider.get().dotProductChunk(tmpValBatch, input,
                                            valueAttnWeights, 0, config.embeddingLength, chunkStart, chunkLength);
                                }
                            }, splitSize, m.getPool()));
                    queryTask.join();
                    keyTask.join();
                    valueTask.join();
                }
            } else {
                qkvResults[0] = queryBatch;
                qkvResults[1] = tmpKeyBatch;
                qkvResults[2] = tmpValBatch;
                try (Timer.Context ignoredQkv = InferenceProfiler.timer(metricRegistry, "causalselfattention.qkv_projection").time()) {
                    VectorMath.pchunk(0, attentionLength, (chunkStart, chunkLength) -> {
                    long start = System.nanoTime();
                    configurableTensorProvider.get()
                            .dotProductBatchChunk(qkvResults, input, qkvWeights, 0, config.embeddingLength, chunkStart, chunkLength);
                    metricRegistry.histogram("causualselfattention.forward_qkv_1").update(System.nanoTime() - start);
                    }, splitSize, m.getPool());
                }
            }

            queryAttnBias.ifPresent(
                    bias -> configurableTensorProvider.get().accumulate(queryBatch, bias,
                            0, attentionLength)
            );
            keyAttnBias.ifPresent(
                    bias -> configurableTensorProvider.get().accumulate(tmpKeyBatch, bias,
                            0, kvLength)
            );
            valueAttnBias.ifPresent(
                    bias -> configurableTensorProvider.get().accumulate(tmpValBatch, bias,
                            0, kvLength)
            );
            m.activeLoraDeltaFor(queryWeightName).ifPresent(
                    delta -> LoraDeltaApplier.apply(m, queryBatch, input, delta));
            m.activeLoraDeltaFor(keyWeightName).ifPresent(
                    delta -> LoraDeltaApplier.apply(m, tmpKeyBatch, input, delta));
            m.activeLoraDeltaFor(valueWeightName).ifPresent(
                    delta -> LoraDeltaApplier.apply(m, tmpValBatch, input, delta));
            normalizeQueryKey(queryBatch, tmpKeyBatch);
            m.emitLayerDebug(layerIndex, "query_projection", queryBatch);
            m.emitLayerDebug(layerIndex, "key_projection", tmpKeyBatch);
            m.emitLayerDebug(layerIndex, "value_projection", tmpValBatch);
            AbstractTensor[] querySlices = new AbstractTensor[batchSize];
            AbstractTensor[] keySlices = new AbstractTensor[batchSize];
            AbstractTensor[] valSlices = new AbstractTensor[batchSize];
            AbstractTensor[] valueSlices = new AbstractTensor[batchSize];

            for(int bi= 0 ; bi <batchSize; bi++) {
                querySlices[bi] = queryBatch.slice(bi);
                keySlices[bi] = tmpKeyBatch.slice(bi);
                valSlices[bi] = tmpValBatch.slice(bi);
                valueSlices[bi] = valueBatch.slice(bi);
            }

            // This is our memory of the key and value vectors for each position
            for (int position = startPosition, bi = 0; position < startPosition + batchSize; position++, bi++) {
                int finalPosition = position;
                AbstractTensor key = kvMem.getKeyTensorForPosition(layerIndex, position);
                AbstractTensor val = kvMem.getValTensorForPosition(layerIndex, position);

                AbstractTensor tmpKey = keySlices[bi];
                AbstractTensor tmpVal = valSlices[bi];
                AbstractTensor query = querySlices[bi];
                AbstractTensor value = valueSlices[bi];

                try (Timer.Context ignoredKv = InferenceProfiler.timer(metricRegistry, "causalselfattention.kv_cache_write").time()) {
                    if (key.dType() != tmpKey.dType()) {
                        try (AbstractTensor tmpKey2 = configurableTensorProvider.get().quantize(tmpKey, key.dType(), 0, kvLength);
                             AbstractTensor tmpVal2 = configurableTensorProvider.get().quantize(tmpVal, val.dType(), 0, kvLength)) {
                            key.copyFrom(tmpKey2, 0, 0, kvLength);
                            val.copyFrom(tmpVal2, 0, 0, kvLength);
                        }
                    } else {
                        key.copyFrom(tmpKey, 0, 0, kvLength);
                        val.copyFrom(tmpVal, 0, 0, kvLength);
                    }
                }

                // apply RoPE if present (accounting for huggingface permutation)
                // https://github.com/huggingface/transformers/blob/d533465150532b0c5de167b574e59f64c68b1154/src/transformers/models/llama/convert_llama_weights_to_hf.py#L114
                try (Timer.Context ignoredRope = InferenceProfiler.timer(metricRegistry, "causalselfattention.rope").time()) {
                    config.ropeFreqs.ifPresent(rf -> {
                    int headPiece = config.headSize / 2;
                    int poffset = finalPosition * headPiece;

                    if (config.isGQA) {
                        rotateRopeHeads(query, numberOfHeads, config.headSize, headPiece, poffset, rf);
                        rotateRopeHeads(key, numberOfKeyValueHeads, config.headSize, headPiece, poffset, rf);
                    } else {
                        // apply RoPE rotation to the q and k vectors for each head
                        for (int h = 0; h < numberOfHeads; h++) {
                            // get the q and k vectors for this head
                            int offset = h * config.headSize;
                            // rotate q and k by the freq theta and freq r
                            for (int i = offset, freqIndex = 0; i < (offset + headPiece); i++, freqIndex++) {
                                float q0 = query.get(0, i);
                                float q1 = query.get(0, i + headPiece); // hf permutation is 0,64,1,65 etc...
                                float k00 = key.get(0, i);
                                float k1 = key.get(0, i + headPiece);
                                float[] f = rf[poffset + freqIndex];
                                float fcr = f[0];
                                float fci = f[1];
                                query.set(q0 * fcr - q1 * fci, 0, i);
                                query.set(q0 * fci + q1 * fcr, 0, i + headPiece);
                                key.set(k00 * fcr - k1 * fci, 0, i);
                                key.set(k00 * fci + k1 * fcr, 0, i + headPiece);
                            }
                        }
                    }
                    debug("query+rope", query, finalPosition);
                    debug("key+rope", key, finalPosition);
                    });
                }

                kvMem.kvRowWritten(layerIndex, finalPosition, kvLength);

                if (!usePackedPrefill(batchSize)) {
                    decodePagedAttention(query, value, kvMem, finalPosition);
                }
            }
            if (usePackedPrefill(batchSize)) {
                prefillAttention(queryBatch, valueBatch, kvMem, startPosition, batchSize);
            }

            // matmul the projection and sum into input
            // input += c_proj_weight @ ybuf + c_proj_bias
            m.emitLayerDebug(layerIndex, "attention_value", valueBatch);
            AbstractTensor result = m.makeDenseTensor(batchSize, config.embeddingLength);
            try (AbstractTensor vq = m.maybeQuantize(valueBatch)) {
                try (Timer.Context ignoredOutput = InferenceProfiler.timer(metricRegistry, "causalselfattention.output_projection").time()) {
                    io.teknek.deliverance.tensor.operations.TensorOperations outputOps =
                            m.prefillProjectionOperations(vq, outputProjectionWeights, phase);
                    VectorMath.pchunk(0, config.embeddingLength, (chunkStart, chunkSize) -> {
                    outputOps.dotProductChunk(
                                     result,
                                     vq,
                                     outputProjectionWeights,
                                    0,
                                    attentionLength,
                                    chunkStart,
                                     chunkSize
                             );
                    }, outputOps.parallelSplitSize(), m.getPool());
                }
                AbstractTensor reduced = m.getTensorParallelContext().enabled()
                        ? allReduceAttention(result)
                        : result;
                m.emitLayerDebug(layerIndex, "attention_output", reduced);
                tensorReducer.ifPresent(func -> func.accept(Collections.singletonList(reduced)));
                outputProjectionBias.ifPresent(bias -> configurableTensorProvider.get().accumulate(reduced, bias, 0, config.embeddingLength));
                // vq is whatever the base o_proj matmul itself consumes; LoraDeltaApplier dequantizes it to
                // match loraA's dtype internally -- see step 4 plan Section 11 item 9 (revised).
                m.activeLoraDeltaFor(outputWeightName).ifPresent(
                        delta -> LoraDeltaApplier.apply(m, reduced, vq, delta));
                if (reduced != result) {
                    result.close();
                }
                return reduced;
            }
        }
        }
    }

    /**
     * Applies RoPE rotation to each attention head stored contiguously in a single row tensor.
     *
     * <p>The tensor layout is {@code [1, heads * headSize]}. Each head is split into two equal halves. For a
     * {@code headSize} of 4, one head is laid out as {@code [x0, x1, y0, y1]}; with {@code cos=0} and {@code sin=1},
     * this rotates to {@code [-y0, -y1, x0, x1]}.</p>
     */
    static void rotateRopeHeads(AbstractTensor tensor, int heads, int headSize, int headPiece, int positionOffset,
            float[][] ropeFreqs) {
        for (int h = 0; h < heads; h++) {
            int offset = h * headSize;
            if (offset >= tensor.shape().last()) {
                break;
            }
            for (int i = offset, freqIndex = 0; i < (offset + headPiece); i++, freqIndex++) {
                float x0 = tensor.get(0, i);
                float x1 = tensor.get(0, i + headPiece);
                float[] f = ropeFreqs[positionOffset + freqIndex];
                float fcr = f[0];
                float fci = f[1];
                tensor.set(x0 * fcr - x1 * fci, 0, i);
                tensor.set(x0 * fci + x1 * fcr, 0, i + headPiece);
            }
        }
    }

    private AbstractTensor allReduceAttention(AbstractTensor result) {
        try (Timer.Context ignored = InferenceProfiler.timer(metricRegistry, "causalselfattention.all_reduce").time()) {
            return m.getTensorParallelCollectives().allReduceSum("layer." + layerIndex + ".self_attn.o_proj", result);
        }
    }

    private boolean usePackedPrefill(int batchSize) {
        return batchSize > 1 && Boolean.parseBoolean(System.getProperty(PACKED_PREFILL_PROPERTY, "true"));
    }

    private void prefillAttention(AbstractTensor queryBatch, AbstractTensor valueBatch, KvBufferCache.KvBuffer kvMem,
            int startPosition, int batchSize) {
        InferenceProfiler.counter(metricRegistry, "causalselfattention.prefill_packed.calls").inc();
        InferenceProfiler.counter(metricRegistry, "causalselfattention.prefill_packed.batch_tokens").inc(batchSize);
        int finalPosition = startPosition + batchSize - 1;
        AbstractTensor[] keyPages = kvMem.getKeyTensorsUptoPosition(layerIndex, finalPosition);
        AbstractTensor[] valuePages = kvMem.getValTensorsUptoPosition(layerIndex, finalPosition);
        recordKvLayout(keyPages, finalPosition + 1);
        try (AbstractTensor packedKeys = m.makeDenseTensor(finalPosition + 1, kvLength);
             AbstractTensor packedValues = m.makeDenseTensor(finalPosition + 1, kvLength)) {
            try (Timer.Context ignoredPack = InferenceProfiler.timer(metricRegistry, "causalselfattention.prefill_pack_kv").time()) {
                fillVisibleRows(packedKeys, keyPages, finalPosition, 0, kvLength);
                fillVisibleRows(packedValues, valuePages, finalPosition, 0, kvLength);
            }
            try (Timer.Context ignoredScore = InferenceProfiler.timer(metricRegistry, "causalselfattention.score_value").time();
                 Timer.Context ignoredPackedScore = InferenceProfiler.timer(metricRegistry, "causalselfattention.prefill_packed.score_value").time()) {
                int headGroupParallelism = Math.min(4, numberOfHeads);
                for (int headStart = 0; headStart < numberOfHeads; headStart += headGroupParallelism) {
                    int headEnd = Math.min(numberOfHeads, headStart + headGroupParallelism);
                    VectorMath.pfor(headStart, headEnd, h -> {
                        int xoffset = Math.floorDiv(h, headGroupSize) * config.headSize;
                        int yoffset = h * config.headSize;
                        if (yoffset >= queryBatch.shape().last()) return;
                        try (AbstractTensor attn = m.makeDenseTensor(batchSize, finalPosition + 1)) {
                            configurableTensorProvider.get()
                                    .batchDotProduct(attn, queryBatch, packedKeys, yoffset, xoffset, config.headSize,
                                            0, 0, finalPosition + 1);
                            for (int bi = 0; bi < batchSize; bi++) {
                                int visibleLength = startPosition + bi + 1;
                                try (AbstractTensor attnRow = attn.slice(bi);
                                     AbstractTensor valueRow = valueBatch.slice(bi)) {
                                    scaledSoftmax(attnRow, visibleLength, attentionScale, config.attnLogitSoftCapping);
                                    configurableTensorProvider.get().saxpy(attnRow, packedValues, valueRow,
                                            xoffset, yoffset, config.headSize, 0, 0, visibleLength);
                                }
                            }
                        }
                    }, m.getPool());
                }
            }
        } finally {
            closeAll(keyPages);
            closeAll(valuePages);
        }
    }

    private void decodePagedAttention(AbstractTensor query, AbstractTensor value, KvBufferCache.KvBuffer kvMem,
            int finalPosition) {
        InferenceProfiler.counter(metricRegistry, "causalselfattention.decode_paged_attention.calls").inc();
        InferenceProfiler.counter(metricRegistry, "causalselfattention.decode_paged_attention.visible_rows")
                .inc(finalPosition + 1L);
        // Old path, kept here while KvPageTable is introduced:
        // AbstractTensor[] keyPages = kvMem.getKeyTensorsUptoPosition(layerIndex, finalPosition);
        // AbstractTensor[] valuePages = kvMem.getValTensorsUptoPosition(layerIndex, finalPosition);
        KvPageTable pageTable = kvMem.getPageTable(layerIndex, finalPosition);
        AbstractTensor[] keyPages = pageTable.keyPages();
        AbstractTensor[] valuePages = pageTable.valuePages();
        recordKvLayout(keyPages, pageTable.visibleRows());
        try {
            try (Timer.Context ignoredScore = InferenceProfiler.timer(metricRegistry, "causalselfattention.score_value").time();
                 Timer.Context ignoredPackedScore = InferenceProfiler.timer(metricRegistry,
                         "causalselfattention.decode_paged_attention").time()) {
                if (m.getDecodeAttentionMode() == DecodeAttentionMode.FLASH_DECODE) {
                    InferenceProfiler.counter(metricRegistry, "causalselfattention.decode_paged_attention.mode_flash_decode").inc();
                    TensorOperations gpuFlashOps = flashDecodeAttentionOperations(value, query, pageTable);
                    if (gpuFlashOps != null) {
                        InferenceProfiler.counter(metricRegistry,
                                "causalselfattention.decode_paged_attention.mode_flash_decode_gpu").inc();
                        gpuFlashOps.flashDecodePagedAttention(value, query, keyPages, valuePages,
                                pageTable.visibleRows(), numberOfHeads, numberOfKeyValueHeads, config.headSize,
                                attentionScale, config.attnLogitSoftCapping);
                    } else {
                        InferenceProfiler.counter(metricRegistry,
                                "causalselfattention.decode_paged_attention.mode_flash_decode_page_block_simd").inc();
                        flashDecodeAttentionPageBlockSimdF32ParallelHeads(value, query, keyPages, valuePages,
                                pageTable.visibleRows(), numberOfHeads, numberOfKeyValueHeads, config.headSize,
                                attentionScale, config.attnLogitSoftCapping, m.getPool());
                    }
                } else {
                    InferenceProfiler.counter(metricRegistry, "causalselfattention.decode_paged_attention.mode_staged").inc();
                    TensorOperations decodeAttentionOps = decodeAttentionOperations(value, query, pageTable);
                    if (InferenceProfiler.isEnabled()) {
                        InferenceProfiler.counter(metricRegistry, "causalselfattention.decode_paged_attention.provider_"
                                + decodeAttentionOps.name().replace(' ', '_')).inc();
                    }
                    decodeAttentionOps.decodePagedAttention(value, query, keyPages, valuePages,
                            pageTable.visibleRows(), numberOfHeads, numberOfKeyValueHeads, config.headSize, attentionScale,
                            config.attnLogitSoftCapping);
                }
            }
        } finally {
            // KvPageTable is cached by KvBuffer and does not own the underlying page views.
        }
    }

    static void flashDecodeAttention(AbstractTensor valueOut, AbstractTensor query, AbstractTensor[] keyPages,
            AbstractTensor[] valuePages, int visibleRows, int numberOfHeads, int numberOfKeyValueHeads, int headSize,
            float scale, Float softcap) {
        Preconditions.checkArgument(keyPages.length == valuePages.length, "key/value page count mismatch");
        Preconditions.checkArgument(query.shape().first() == 1, "decode query must have one row");
        Preconditions.checkArgument(valueOut.shape().first() == 1, "decode value output must have one row");
        Preconditions.checkArgument(visibleRows > 0, "visibleRows must be positive");
        int headGroupSize = numberOfHeads / numberOfKeyValueHeads;
        for (int head = 0; head < numberOfHeads; head++) {
            flashDecodeAttentionHead(valueOut, query, keyPages, valuePages, visibleRows, headGroupSize, headSize,
                    scale, softcap, head);
        }
    }

    static void flashDecodeAttentionParallelHeads(AbstractTensor valueOut, AbstractTensor query,
            AbstractTensor[] keyPages, AbstractTensor[] valuePages, int visibleRows, int numberOfHeads,
            int numberOfKeyValueHeads, int headSize, float scale, Float softcap, WrappedForkJoinPool pool) {
        Preconditions.checkArgument(keyPages.length == valuePages.length, "key/value page count mismatch");
        Preconditions.checkArgument(query.shape().first() == 1, "decode query must have one row");
        Preconditions.checkArgument(valueOut.shape().first() == 1, "decode value output must have one row");
        Preconditions.checkArgument(visibleRows > 0, "visibleRows must be positive");
        int headGroupSize = numberOfHeads / numberOfKeyValueHeads;
        VectorMath.pfor(0, numberOfHeads, head -> flashDecodeAttentionHead(valueOut, query, keyPages, valuePages,
                visibleRows, headGroupSize, headSize, scale, softcap, head), pool);
    }

    static void flashDecodeAttentionSimdF32ParallelHeads(AbstractTensor valueOut, AbstractTensor query,
            AbstractTensor[] keyPages, AbstractTensor[] valuePages, int visibleRows, int numberOfHeads,
            int numberOfKeyValueHeads, int headSize, float scale, Float softcap, WrappedForkJoinPool pool) {
        Preconditions.checkArgument(keyPages.length == valuePages.length, "key/value page count mismatch");
        Preconditions.checkArgument(query.shape().first() == 1, "decode query must have one row");
        Preconditions.checkArgument(valueOut.shape().first() == 1, "decode value output must have one row");
        Preconditions.checkArgument(visibleRows > 0, "visibleRows must be positive");
        Preconditions.checkArgument(valueOut.shape().last() >= numberOfHeads * headSize,
                "valueOut width must contain all query heads");
        Preconditions.checkArgument(query.shape().last() >= numberOfHeads * headSize,
                "query width must contain all query heads");
        int kvLength = numberOfKeyValueHeads * headSize;
        for (int pageIndex = 0; pageIndex < keyPages.length; pageIndex++) {
            Preconditions.checkArgument(keyPages[pageIndex].shape().last() >= kvLength,
                    "key page width must contain all KV heads");
            Preconditions.checkArgument(valuePages[pageIndex].shape().last() >= kvLength,
                    "value page width must contain all KV heads");
        }
        if (!(valueOut instanceof FloatBufferTensor valueOutF32) || !(query instanceof FloatBufferTensor queryF32)
                || !allFloatBufferTensors(keyPages) || !allFloatBufferTensors(valuePages)) {
            flashDecodeAttentionParallelHeads(valueOut, query, keyPages, valuePages, visibleRows, numberOfHeads,
                    numberOfKeyValueHeads, headSize, scale, softcap, pool);
            return;
        }
        int headGroupSize = numberOfHeads / numberOfKeyValueHeads;
        VectorMath.pfor(0, numberOfHeads, head -> flashDecodeAttentionHeadSimdF32(valueOutF32, queryF32, keyPages,
                valuePages, visibleRows, headGroupSize, headSize, scale, softcap, head), pool);
    }

    static void flashDecodeAttentionTensorPlanParallelHeads(TensorPlan plan, AbstractTensor valueOut,
            AbstractTensor query, AbstractTensor[] keyPages, AbstractTensor[] valuePages, int visibleRows,
            int numberOfHeads, int numberOfKeyValueHeads, int headSize, float scale, Float softcap,
            WrappedForkJoinPool pool) {
        Preconditions.checkArgument(keyPages.length == valuePages.length, "key/value page count mismatch");
        Preconditions.checkArgument(query.shape().first() == 1, "decode query must have one row");
        Preconditions.checkArgument(valueOut.shape().first() == 1, "decode value output must have one row");
        Preconditions.checkArgument(visibleRows > 0, "visibleRows must be positive");
        Preconditions.checkArgument(valueOut.shape().last() >= numberOfHeads * headSize,
                "valueOut width must contain all query heads");
        Preconditions.checkArgument(query.shape().last() >= numberOfHeads * headSize,
                "query width must contain all query heads");
        int kvLength = numberOfKeyValueHeads * headSize;
        for (int pageIndex = 0; pageIndex < keyPages.length; pageIndex++) {
            Preconditions.checkArgument(keyPages[pageIndex].shape().last() >= kvLength,
                    "key page width must contain all KV heads");
            Preconditions.checkArgument(valuePages[pageIndex].shape().last() >= kvLength,
                    "value page width must contain all KV heads");
        }
        int headGroupSize = numberOfHeads / numberOfKeyValueHeads;
        VectorMath.pfor(0, numberOfHeads, head -> flashDecodeAttentionHeadTensorPlan(plan, valueOut, query, keyPages,
                valuePages, visibleRows, headGroupSize, headSize, scale, softcap, head), pool);
    }

    static void flashDecodeAttentionPageBlockSimdF32ParallelHeads(AbstractTensor valueOut, AbstractTensor query,
            AbstractTensor[] keyPages, AbstractTensor[] valuePages, int visibleRows, int numberOfHeads,
            int numberOfKeyValueHeads, int headSize, float scale, Float softcap, WrappedForkJoinPool pool) {
        Preconditions.checkArgument(keyPages.length == valuePages.length, "key/value page count mismatch");
        Preconditions.checkArgument(query.shape().first() == 1, "decode query must have one row");
        Preconditions.checkArgument(valueOut.shape().first() == 1, "decode value output must have one row");
        Preconditions.checkArgument(visibleRows > 0, "visibleRows must be positive");
        Preconditions.checkArgument(valueOut.shape().last() >= numberOfHeads * headSize,
                "valueOut width must contain all query heads");
        Preconditions.checkArgument(query.shape().last() >= numberOfHeads * headSize,
                "query width must contain all query heads");
        int kvLength = numberOfKeyValueHeads * headSize;
        for (int pageIndex = 0; pageIndex < keyPages.length; pageIndex++) {
            Preconditions.checkArgument(keyPages[pageIndex].shape().last() >= kvLength,
                    "key page width must contain all KV heads");
            Preconditions.checkArgument(valuePages[pageIndex].shape().last() >= kvLength,
                    "value page width must contain all KV heads");
        }
        if (!(valueOut instanceof FloatBufferTensor valueOutF32) || !(query instanceof FloatBufferTensor queryF32)
                || !allFloatBufferTensors(keyPages) || !allFloatBufferTensors(valuePages)) {
            flashDecodeAttentionParallelHeads(valueOut, query, keyPages, valuePages, visibleRows, numberOfHeads,
                    numberOfKeyValueHeads, headSize, scale, softcap, pool);
            return;
        }
        int headGroupSize = numberOfHeads / numberOfKeyValueHeads;
        VectorMath.pfor(0, numberOfHeads, head -> flashDecodeAttentionHeadPageBlockSimdF32(valueOutF32, queryF32,
                keyPages, valuePages, visibleRows, headGroupSize, headSize, scale, softcap, head), pool);
    }

    private static void flashDecodeAttentionHeadPageBlockSimdF32(FloatBufferTensor valueOut, FloatBufferTensor query,
            AbstractTensor[] keyPages, AbstractTensor[] valuePages, int visibleRows, int headGroupSize, int headSize,
            float scale, Float softcap, int head) {
        int kvHead = head / headGroupSize;
        int queryOffset = head * headSize;
        int kvOffset = kvHead * headSize;
        zeroSliceF32(valueOut, queryOffset, headSize);

        int maxPageRows = 0;
        for (AbstractTensor keyPage : keyPages) {
            maxPageRows = Math.max(maxPageRows, keyPage.shape().first());
        }
        float[] scores = new float[maxPageRows];
        double runningMax = Double.NEGATIVE_INFINITY;
        double runningDenom = 0.0;
        int globalRow = 0;
        for (int pageIndex = 0; pageIndex < keyPages.length && globalRow < visibleRows; pageIndex++) {
            FloatBufferTensor keyPage = (FloatBufferTensor) keyPages[pageIndex];
            FloatBufferTensor valuePage = (FloatBufferTensor) valuePages[pageIndex];
            int rows = Math.min(keyPage.shape().first(), visibleRows - globalRow);
            double pageMax = Double.NEGATIVE_INFINITY;
            for (int row = 0; row < rows; row++) {
                float dot = dotHeadF32(query, queryOffset, keyPage, row, kvOffset, headSize);
                scores[row] = transformForAttentionSoftmax(dot, scale, softcap);
                pageMax = Math.max(pageMax, scores[row]);
            }
            double newMax = Math.max(runningMax, pageMax);
            float oldScale = runningMax == Double.NEGATIVE_INFINITY ? 0.0f
                    : (float) FastMath.exp(runningMax - newMax);
            double pageDenom = 0.0;
            for (int row = 0; row < rows; row++) {
                float weight = (float) FastMath.exp(scores[row] - newMax);
                pageDenom += weight;
                if (row == 0) {
                    rescaleAndAccumulateF32(valueOut, queryOffset, valuePage, row, kvOffset, headSize, oldScale,
                            weight);
                } else {
                    accumulateWeightedF32(valueOut, queryOffset, valuePage, row, kvOffset, headSize, weight);
                }
            }
            runningDenom = runningDenom * oldScale + pageDenom;
            runningMax = newMax;
            globalRow += rows;
        }
        normalizeSliceF32(valueOut, queryOffset, headSize, (float) (1.0 / runningDenom));
    }

    private static void flashDecodeAttentionHeadTensorPlan(TensorPlan plan, AbstractTensor valueOut,
            AbstractTensor query, AbstractTensor[] keyPages, AbstractTensor[] valuePages, int visibleRows,
            int headGroupSize, int headSize, float scale, Float softcap, int head) {
        int kvHead = head / headGroupSize;
        int queryOffset = head * headSize;
        int kvOffset = kvHead * headSize;
        for (int col = 0; col < headSize; col++) {
            valueOut.set(0.0f, 0, queryOffset + col);
        }

        int maxPageRows = 0;
        for (AbstractTensor keyPage : keyPages) {
            maxPageRows = Math.max(maxPageRows, keyPage.shape().first());
        }
        float[] scores = new float[maxPageRows];
        double runningMax = Double.NEGATIVE_INFINITY;
        double runningDenom = 0.0;
        int globalRow = 0;
        for (int pageIndex = 0; pageIndex < keyPages.length && globalRow < visibleRows; pageIndex++) {
            AbstractTensor keyPage = keyPages[pageIndex];
            AbstractTensor valuePage = valuePages[pageIndex];
            int rows = Math.min(keyPage.shape().first(), visibleRows - globalRow);
            plan.dotRowsToArray(query, 0, queryOffset, keyPage, 0, kvOffset, rows, headSize, scores, 0);
            double pageMax = Double.NEGATIVE_INFINITY;
            for (int row = 0; row < rows; row++) {
                scores[row] = transformForAttentionSoftmax(scores[row], scale, softcap);
                pageMax = Math.max(pageMax, scores[row]);
            }
            double newMax = Math.max(runningMax, pageMax);
            float oldScale = runningMax == Double.NEGATIVE_INFINITY ? 0.0f
                    : (float) FastMath.exp(runningMax - newMax);
            double pageDenom = 0.0;
            for (int row = 0; row < rows; row++) {
                scores[row] = (float) FastMath.exp(scores[row] - newMax);
                pageDenom += scores[row];
            }
            if (oldScale != 1.0f) {
                plan.scaleSlice(valueOut, 0, queryOffset, headSize, oldScale);
            }
            plan.accumulateWeightedRows(valueOut, 0, queryOffset, valuePage, 0, kvOffset, rows, headSize, scores, 0);
            runningDenom = runningDenom * oldScale + pageDenom;
            runningMax = newMax;
            globalRow += rows;
        }
        plan.normalizeSlice(valueOut, 0, queryOffset, headSize, (float) (1.0 / runningDenom));
    }

    private static boolean allFloatBufferTensors(AbstractTensor[] tensors) {
        for (AbstractTensor tensor : tensors) {
            if (!(tensor instanceof FloatBufferTensor)) {
                return false;
            }
        }
        return true;
    }

    private static void flashDecodeAttentionHeadSimdF32(FloatBufferTensor valueOut, FloatBufferTensor query,
            AbstractTensor[] keyPages, AbstractTensor[] valuePages, int visibleRows, int headGroupSize, int headSize,
            float scale, Float softcap, int head) {
        int kvHead = head / headGroupSize;
        int queryOffset = head * headSize;
        int kvOffset = kvHead * headSize;
        zeroSliceF32(valueOut, queryOffset, headSize);

        double runningMax = Double.NEGATIVE_INFINITY;
        double runningDenom = 0.0;
        int globalRow = 0;
        for (int pageIndex = 0; pageIndex < keyPages.length && globalRow < visibleRows; pageIndex++) {
            FloatBufferTensor keyPage = (FloatBufferTensor) keyPages[pageIndex];
            FloatBufferTensor valuePage = (FloatBufferTensor) valuePages[pageIndex];
            int rows = Math.min(keyPage.shape().first(), visibleRows - globalRow);
            for (int row = 0; row < rows; row++) {
                float dot = dotHeadF32(query, queryOffset, keyPage, row, kvOffset, headSize);
                double score = transformForAttentionSoftmax(dot, scale, softcap);
                double newMax = Math.max(runningMax, score);
                float oldScale = runningMax == Double.NEGATIVE_INFINITY ? 0.0f
                        : (float) FastMath.exp(runningMax - newMax);
                float weight = (float) FastMath.exp(score - newMax);
                rescaleAndAccumulateF32(valueOut, queryOffset, valuePage, row, kvOffset, headSize, oldScale, weight);
                runningDenom = runningDenom * oldScale + weight;
                runningMax = newMax;
            }
            globalRow += rows;
        }
        normalizeSliceF32(valueOut, queryOffset, headSize, (float) (1.0 / runningDenom));
    }

    private static float dotHeadF32(FloatBufferTensor query, int queryOffset, FloatBufferTensor keyPage, int keyRow,
            int keyOffset, int headSize) {
        FloatVector acc = FloatVector.zero(F32_SPECIES);
        int upper = F32_SPECIES.loopBound(headSize);
        int col = 0;
        for (; col < upper; col += F32_SPECIES.length()) {
            acc = query.getVector(F32_SPECIES, 0, queryOffset + col)
                    .fma(keyPage.getVector(F32_SPECIES, keyRow, keyOffset + col), acc);
        }
        float dot = acc.reduceLanes(VectorOperators.ADD);
        for (; col < headSize; col++) {
            dot += query.get(0, queryOffset + col) * keyPage.get(keyRow, keyOffset + col);
        }
        return dot;
    }

    private static void zeroSliceF32(FloatBufferTensor valueOut, int offset, int length) {
        FloatVector zero = FloatVector.zero(F32_SPECIES);
        int upper = F32_SPECIES.loopBound(length);
        int col = 0;
        for (; col < upper; col += F32_SPECIES.length()) {
            valueOut.intoTensor(zero, 0, offset + col);
        }
        for (; col < length; col++) {
            valueOut.set(0.0f, 0, offset + col);
        }
    }

    private static void rescaleAndAccumulateF32(FloatBufferTensor valueOut, int outOffset,
            FloatBufferTensor valuePage, int valueRow, int valueOffset, int length, float oldScale, float weight) {
        FloatVector oldScaleVector = FloatVector.broadcast(F32_SPECIES, oldScale);
        FloatVector weightVector = FloatVector.broadcast(F32_SPECIES, weight);
        int upper = F32_SPECIES.loopBound(length);
        int col = 0;
        for (; col < upper; col += F32_SPECIES.length()) {
            FloatVector out = valueOut.getVector(F32_SPECIES, 0, outOffset + col);
            FloatVector value = valuePage.getVector(F32_SPECIES, valueRow, valueOffset + col);
            valueOut.intoTensor(value.fma(weightVector, out.mul(oldScaleVector)), 0, outOffset + col);
        }
        for (; col < length; col++) {
            valueOut.set(valueOut.get(0, outOffset + col) * oldScale
                    + valuePage.get(valueRow, valueOffset + col) * weight, 0, outOffset + col);
        }
    }

    private static void accumulateWeightedF32(FloatBufferTensor valueOut, int outOffset, FloatBufferTensor valuePage,
            int valueRow, int valueOffset, int length, float weight) {
        FloatVector weightVector = FloatVector.broadcast(F32_SPECIES, weight);
        int upper = F32_SPECIES.loopBound(length);
        int col = 0;
        for (; col < upper; col += F32_SPECIES.length()) {
            FloatVector out = valueOut.getVector(F32_SPECIES, 0, outOffset + col);
            FloatVector value = valuePage.getVector(F32_SPECIES, valueRow, valueOffset + col);
            valueOut.intoTensor(value.fma(weightVector, out), 0, outOffset + col);
        }
        for (; col < length; col++) {
            valueOut.set(valueOut.get(0, outOffset + col) + valuePage.get(valueRow, valueOffset + col) * weight,
                    0, outOffset + col);
        }
    }

    private static void normalizeSliceF32(FloatBufferTensor valueOut, int offset, int length, float invDenom) {
        FloatVector inv = FloatVector.broadcast(F32_SPECIES, invDenom);
        int upper = F32_SPECIES.loopBound(length);
        int col = 0;
        for (; col < upper; col += F32_SPECIES.length()) {
            valueOut.intoTensor(valueOut.getVector(F32_SPECIES, 0, offset + col).mul(inv), 0, offset + col);
        }
        for (; col < length; col++) {
            valueOut.set(valueOut.get(0, offset + col) * invDenom, 0, offset + col);
        }
    }

    private static void flashDecodeAttentionHead(AbstractTensor valueOut, AbstractTensor query,
            AbstractTensor[] keyPages, AbstractTensor[] valuePages, int visibleRows, int headGroupSize, int headSize,
            float scale, Float softcap, int head) {
        int kvHead = head / headGroupSize;
        int queryOffset = head * headSize;
        int kvOffset = kvHead * headSize;
        for (int col = 0; col < headSize; col++) {
            valueOut.set(0.0f, 0, queryOffset + col);
        }

        double runningMax = Double.NEGATIVE_INFINITY;
        double runningDenom = 0.0;
        int globalRow = 0;
        for (int pageIndex = 0; pageIndex < keyPages.length && globalRow < visibleRows; pageIndex++) {
            AbstractTensor keyPage = keyPages[pageIndex];
            AbstractTensor valuePage = valuePages[pageIndex];
            int rows = Math.min(keyPage.shape().first(), visibleRows - globalRow);
            for (int row = 0; row < rows; row++) {
                float dot = 0.0f;
                for (int col = 0; col < headSize; col++) {
                    dot += query.get(0, queryOffset + col) * keyPage.get(row, kvOffset + col);
                }
                double score = transformForAttentionSoftmax(dot, scale, softcap);
                double newMax = Math.max(runningMax, score);
                double oldScale = runningMax == Double.NEGATIVE_INFINITY ? 0.0 : FastMath.exp(runningMax - newMax);
                double weight = FastMath.exp(score - newMax);
                for (int col = 0; col < headSize; col++) {
                    double accumulated = valueOut.get(0, queryOffset + col) * oldScale
                            + weight * valuePage.get(row, kvOffset + col);
                    valueOut.set((float) accumulated, 0, queryOffset + col);
                }
                runningDenom = runningDenom * oldScale + weight;
                runningMax = newMax;
            }
            globalRow += rows;
        }

        float invDenom = (float) (1.0 / runningDenom);
        for (int col = 0; col < headSize; col++) {
            valueOut.set(valueOut.get(0, queryOffset + col) * invDenom, 0, queryOffset + col);
        }
    }

    private static float transformForAttentionSoftmax(float value, float scale, Float softcap) {
        float scaled = value * scale;
        if (softcap == null) {
            return scaled;
        }
        return (float) FastMath.tanh(scaled / softcap) * softcap;
    }

    private TensorOperations decodeAttentionOperations(AbstractTensor valueOut, AbstractTensor query,
            KvPageTable pageTable) {
        TensorOperations primary = configurableTensorProvider.get();
        if (m.isGpuDecodeAttentionEnabled() && !m.isTensorProviderExplicit()) {
            Optional<TensorOperations> gpu = m.tensorOperations(TensorProviderKind.GPU);
            if (gpu.isPresent() && gpu.get().supportsDecodePagedAttention(valueOut, query, pageTable.keyPages(),
                    pageTable.valuePages(), pageTable.visibleRows(), numberOfHeads, numberOfKeyValueHeads, config.headSize, attentionScale,
                    config.attnLogitSoftCapping)) {
                return gpu.get();
            }
        }
        return primary;
    }

    private TensorOperations flashDecodeAttentionOperations(AbstractTensor valueOut, AbstractTensor query,
            KvPageTable pageTable) {
        if (!m.isGpuFlashDecodeAttentionEnabled() || m.isTensorProviderExplicit()) {
            return null;
        }
        Optional<TensorOperations> gpu = m.tensorOperations(TensorProviderKind.GPU);
        if (gpu.isPresent() && gpu.get().supportsFlashDecodePagedAttention(valueOut, query, pageTable.keyPages(),
                pageTable.valuePages(), pageTable.visibleRows(), numberOfHeads, numberOfKeyValueHeads,
                config.headSize, attentionScale, config.attnLogitSoftCapping)) {
            return gpu.get();
        }
        return null;
    }

    private void recordKvLayout(AbstractTensor[] kvp, int visibleRows) {
        if (!InferenceProfiler.isEnabled() || kvp.length == 0) {
            return;
        }
        InferenceProfiler.counter(metricRegistry, "causalselfattention.kvpages_total").inc(kvp.length);
        InferenceProfiler.counter(metricRegistry, "causalselfattention.kvpages_calls").inc();
        InferenceProfiler.counter(metricRegistry, "causalselfattention.kvpages_" + bucket(kvp.length)).inc();
        InferenceProfiler.counter(metricRegistry, "causalselfattention.kvrows_visible_total").inc(visibleRows);
        InferenceProfiler.counter(metricRegistry, "causalselfattention.kvrows_visible_" + bucket(visibleRows)).inc();

        int fullPageRows = kvp[0].shape().first();
        int finalPageRows = visibleRows - (fullPageRows * (kvp.length - 1));
        InferenceProfiler.counter(metricRegistry, "causalselfattention.kvrows_capacity_total").inc(fullPageRows);
        InferenceProfiler.counter(metricRegistry, "causalselfattention.kvrows_final_total").inc(finalPageRows);
        InferenceProfiler.counter(metricRegistry, "causalselfattention.kvrows_final_" + bucket(finalPageRows)).inc();
    }

    private String bucket(int value) {
        if (value <= 1) {
            return "1";
        }
        int upper = Integer.highestOneBit(value - 1) << 1;
        return "le_" + upper;
    }

    /**
     * Optional family-specific hook applied after Q/K/V projection and bias, before RoPE and KV-cache writes.
     *
     * <p>Most older decoder families handled by {@link CausalSelfAttention} do not apply a separate normalization to
     * projected query/key heads, so the default implementation is intentionally a no-op. Newer families such as Qwen3
     * apply RMSNorm independently to each query and key head after projection:</p>
     *
     * <pre>{@code
     * query = q_norm(q_proj(hidden_states).view(...))
     * key   = k_norm(k_proj(hidden_states).view(...))
     * query, key = apply_rope(query, key)
     * }</pre>
     *
     * <p>Subclasses should override this method when the upstream architecture specifies Q/K normalization at this
     * point in the attention pipeline. Implementations must mutate {@code queryBatch} and {@code keyBatch} in place,
     * preserve their shapes, and must not close either tensor because ownership remains with the caller.</p>
     */
    protected void normalizeQueryKey(AbstractTensor queryBatch, AbstractTensor keyBatch) {
    }
}
