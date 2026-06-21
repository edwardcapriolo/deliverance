package io.teknek.deliverance.generator;

import com.codahale.metrics.MetricRegistry;
import com.codahale.metrics.Timer;
import com.google.common.base.Preconditions;
import io.teknek.deliverance.math.VectorMath;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.KvBufferCache;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.function.Consumer;

import static io.teknek.deliverance.tensor.DebugSupport.debug;

public class CausalSelfAttention extends BaseCausalSelfAttention {
    private static final Logger logger = LoggerFactory.getLogger(CausalSelfAttention.class);

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
                Optional.empty(),
                Optional.empty(),
                Optional.empty(),
                queryAttnWeights,
                keyAttnWeights,
                valueAttnWeights,
                Optional.empty(),
                outputProjectionWeights,
                configurableTensorProvider,
                metricRegistry
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
    }

    public AbstractTensor forward(AbstractTensor input, int startPosition, KvBufferCache.KvBuffer kvMem,
            Optional<Consumer<List<AbstractTensor>>> tensorReducer) {
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
                    VectorMath.pchunkMetrics(0, attentionLength, (chunkStart, chunkLength) -> {
                    configurableTensorProvider.get()
                                .dotProductChunk(queryBatch, input, queryAttnWeights, 0, config.embeddingLength, chunkStart, chunkLength);
                    }, splitSize, tm, m.getPool());
                    VectorMath.pchunk(0, kvLength, (chunkStart, chunkLength) -> {
                    Timer t = metricRegistry.timer("causualselfattention.forward_gqa_key_2");
                    try (Timer.Context context = t.time()) {
                        configurableTensorProvider.get()
                                .dotProductChunk(tmpKeyBatch, input, keyAttnWeights, 0, config.embeddingLength, chunkStart, chunkLength);
                        context.stop();
                    }
                    Timer r = metricRegistry.timer("causualselfattention.forward_gqa_val_3");
                    try (Timer.Context context = r.time()) {
                        configurableTensorProvider.get()
                                .dotProductChunk(tmpValBatch, input, valueAttnWeights, 0, config.embeddingLength, chunkStart, chunkLength);
                        context.stop();
                    }
                    }, splitSize, m.getPool());
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

                AbstractTensor[] kvp = kvMem.getKeyTensorsUptoPosition(layerIndex, position);
                AbstractTensor[] vvp = kvMem.getValTensorsUptoPosition(layerIndex, position);

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
                        // apply RoPE rotation to the q and k vectors for each head
                        for (int h = 0; h < numberOfHeads; h++) {
                            // get the q vectors for this head
                            int offset = h * config.headSize;

                            // skip if we are out of bounds
                            if (offset >= query.shape().last()) break;

                            int globalHead = m.getTensorParallelContext().rank() * numberOfHeads + h;
                            int goffset = Math.floorDiv(globalHead, headGroupSize) * config.headSize;
                            // rotate q by the freq theta and freq r
                            for (int i = offset, g = goffset; i < (offset + headPiece); i++, g++) {
                                float q0 = query.get(0, i);
                                float q1 = query.get(0, i + headPiece); // hf permutation is 0,64,1,65 etc...
                                float[] f = rf[poffset + g];
                                float fcr = f[0];
                                float fci = f[1];
                                query.set(q0 * fcr - q1 * fci, 0, i);
                                query.set(q0 * fci + q1 * fcr, 0, i + headPiece);
                            }
                        }

                        for (int h = 0; h < numberOfKeyValueHeads; h++) {
                            // get the k vectors for this head
                            int offset = h * config.headSize;
                            if (offset >= key.shape().last()) break;
                            int globalOffset = (m.getTensorParallelContext().rank() * numberOfKeyValueHeads + h)
                                    * config.headSize;
                            // rotate k by the freq theta and freq r
                            for (int i = offset, g = globalOffset; i < (offset + headPiece); i++, g++) {
                                float k00 = key.get(0, i);
                                float k1 = key.get(0, i + headPiece); // hf permutation is 0,64,1,65 etc...
                                float[] f = rf[poffset + g];
                                float fcr = f[0];
                                float fci = f[1];
                                key.set(k00 * fcr - k1 * fci, 0, i);
                                key.set(k00 * fci + k1 * fcr, 0, i + headPiece);
                            }
                        }
                    } else {
                        // apply RoPE rotation to the q and k vectors for each head
                        for (int h = 0; h < numberOfHeads; h++) {
                            // get the q and k vectors for this head
                            int offset = h * config.headSize;
                            // rotate q and k by the freq theta and freq r
                            for (int i = offset; i < (offset + headPiece); i++) {
                                float q0 = query.get(0, i);
                                float q1 = query.get(0, i + headPiece); // hf permutation is 0,64,1,65 etc...
                                float k00 = key.get(0, i);
                                float k1 = key.get(0, i + headPiece);
                                float[] f = rf[poffset + i];
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

                // Attention
                try (Timer.Context ignoredScore = InferenceProfiler.timer(metricRegistry, "causalselfattention.score_value").time()) {
                    VectorMath.pfor(0, numberOfHeads, h -> {
                    int xoffset = Math.floorDiv(h, headGroupSize) * config.headSize;
                    int yoffset = h * config.headSize;

                    if (yoffset >= query.shape().last()) return;

                    try (AbstractTensor attn = m.makeDenseTensor(1, kvp[0].shape().first() * kvp.length)) { // chunky so the cache isn't
                        // thrashed
                        // compute attention scores by multiplying query and key for every position
                        // do this for each position since the pages are not contiguous
                        for (int i = 0; i < kvp.length; i++) {
                            int len = kvp[i].shape().first();
                            int offset = i * len;
                            int size = i == kvp.length - 1 ? (finalPosition + 1) - offset : len;
                            configurableTensorProvider.get()
                                    .batchDotProduct(attn, query, kvp[i], yoffset, xoffset, config.headSize, offset, 0, size);
                        }

                        configurableTensorProvider.get().scale(attentionScale, attn, 0, finalPosition + 1);

                        applyAttentionSoftcap(attn, finalPosition + 1, config.attnLogitSoftCapping);

                        // softmax the scores to get attention weights, from 0..pos inclusively
                        softmax(attn, finalPosition + 1);

                        // apply adjusted attention weights to value vectors
                        // do this for each position since the pages are not contiguous
                        for (int i = 0; i < vvp.length; i++) {
                            int len = vvp[i].shape().first(); // batch size
                            int offset = i * len;
                            int size = i == vvp.length - 1 ? (finalPosition + 1) - offset : len;
                            configurableTensorProvider.get().saxpy(attn, vvp[i], value, xoffset, yoffset, config.headSize, offset, 0, size);
                        }
                    }
                    }, m.getPool());
                }
            }

            // matmul the projection and sum into input
            // input += c_proj_weight @ ybuf + c_proj_bias
            m.emitLayerDebug(layerIndex, "attention_value", valueBatch);
            AbstractTensor result = m.makeDenseTensor(batchSize, config.embeddingLength);
            try (AbstractTensor vq = m.maybeQuantize(valueBatch)) {
                try (Timer.Context ignoredOutput = InferenceProfiler.timer(metricRegistry, "causalselfattention.output_projection").time()) {
                    VectorMath.pchunk(0, config.embeddingLength, (chunkStart, chunkSize) -> {
                    configurableTensorProvider.get()
                            .dotProductChunk(
                                    result,
                                    vq,
                                    outputProjectionWeights,
                                    0,
                                    attentionLength,
                                    chunkStart,
                                    chunkSize
                            );
                    }, splitSize, m.getPool());
                }
                AbstractTensor reduced = m.getTensorParallelContext().enabled()
                        ? allReduceAttention(result)
                        : result;
                m.emitLayerDebug(layerIndex, "attention_output", reduced);
                tensorReducer.ifPresent(func -> func.accept(Collections.singletonList(reduced)));
                outputProjectionBias.ifPresent(bias -> configurableTensorProvider.get().accumulate(reduced, bias, 0, config.embeddingLength));
                if (reduced != result) {
                    result.close();
                }
                return reduced;
            }
        }
        }
    }

    private AbstractTensor allReduceAttention(AbstractTensor result) {
        try (Timer.Context ignored = InferenceProfiler.timer(metricRegistry, "causalselfattention.all_reduce").time()) {
            return m.getTensorParallelCollectives().allReduceSum("layer." + layerIndex + ".self_attn.o_proj", result);
        }
    }
}
