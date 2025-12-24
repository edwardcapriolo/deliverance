package io.teknek.deliverance.generator;

import com.google.common.base.Preconditions;
import io.teknek.deliverance.math.VectorMath;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.safetensors.DistributedContext;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.KvBufferCache;
import io.teknek.deliverance.tensor.VectorTensorMathUtils;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;

import net.jafama.FastMath;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.function.Consumer;

import static io.teknek.deliverance.tensor.DebugSupport.debug;

public class CausalSelfAttention {
    private static final Logger logger = LoggerFactory.getLogger(CausalSelfAttention.class);

    private final AbstractModel m;
    private final Config config;
    private final int layerIndex;
    private final DistributedContext dctx;
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

    private final AbstractTensor[] qkvResults;
    private final AbstractTensor[] qkvWeights;
    private final ConfigurableTensorProvider configurableTensorProvider;

    public CausalSelfAttention(
            AbstractModel m,
            int layerIndex,
            AbstractTensor queryAttnWeights,
            AbstractTensor keyAttnWeights,
            AbstractTensor valueAttnWeights,
            AbstractTensor outputProjectionWeights,
            ConfigurableTensorProvider configurableTensorProvider
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
                configurableTensorProvider
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
            ConfigurableTensorProvider configurableTensorProvider
    ) {
        this.m = m;
        this.layerIndex = layerIndex;
        this.config = m.getConfig();
        this.dctx = m.getConfig().dctx();
        this.queryAttnBias = queryAttnBias;
        this.keyAttnBias = keyAttnBias;
        this.valueAttnBias = valueAttnBias;
        this.queryAttnWeights = queryAttnWeights;
        this.keyAttnWeights = keyAttnWeights;
        this.valueAttnWeights = valueAttnWeights;

        this.outputProjectionBias = outputProjectionBias;
        this.outputProjectionWeights = outputProjectionWeights;
        this.attentionLength = config.numberOfHeads * config.headSize;

        this.attentionScale = config.attentionMultiplier != null ? config.attentionMultiplier : (float) (1.0 / StrictMath.sqrt(config.headSize));

        this.qkvResults = new AbstractTensor[3];
        this.qkvWeights = new AbstractTensor[] { queryAttnWeights, keyAttnWeights, valueAttnWeights };
        this.configurableTensorProvider = configurableTensorProvider;

        configurableTensorProvider.get().registerModelTensor(queryAttnWeights);
        configurableTensorProvider.get().registerModelTensor(keyAttnWeights);
        configurableTensorProvider.get().registerModelTensor(valueAttnWeights);
        configurableTensorProvider.get().registerModelTensor(outputProjectionWeights);
    }

    public AbstractTensor forward(AbstractTensor input, int startPosition, KvBufferCache.KvBuffer kvMem,
            Optional<Consumer<List<AbstractTensor>>> tensorReducer) {
        Preconditions.checkArgument(input.dims() == 2 && input.shape().last() == config.embeddingLength);
        int batchSize = input.shape().first();
        int splitSize = configurableTensorProvider.get().parallelSplitSize();
        try (AbstractTensor queryBatch = m.makeDenseTensor(batchSize, attentionLength);
                AbstractTensor tmpKeyBatch = m.makeDenseTensor(batchSize, config.kvLength);
                AbstractTensor tmpValBatch = m.makeDenseTensor(batchSize, config.kvLength);
                AbstractTensor valueBatch = m.makeDenseTensor(batchSize, attentionLength)) {

            if (config.isGQA) {
                /*
                DistributedContext{c=io.teknek.deliverance.model.llama.LlamaConfig@5df417a7, modelShard=0,
                numModelShards=1, layerShard=0, numLayerShards=1, embeddingSegmentStart=0, embeddingSegmentLength=2048,
                embeddingSegmentEnd=2048, attentionSegmentStart=0, attentionSegmentLength=2048, attentionSegmentEnd=2048,
                hiddenSegmentStart=0, hiddenSegmentLength=5632, hiddenSegmentEnd=5632, kvSegmentStart=0, kvSegmentLength=256,
                kvSegmentEnd=256, headStart=0, headEnd=32, groupHeadStart=0, groupHeadEnd=4, numberOfLayers=22, layerStart=0, layerEnd=22}
                 */
                VectorMath.pchunk(dctx.attentionSegmentStart, dctx.attentionSegmentLength, (chunkStart, chunkLength) -> {
                    configurableTensorProvider.get()
                            .dotProductChunk(queryBatch, input, queryAttnWeights, 0, config.embeddingLength, chunkStart, chunkLength);
                }, splitSize);
                VectorMath.pchunk(dctx.kvSegmentStart, dctx.kvSegmentLength, (chunkStart, chunkLength) -> {
                    configurableTensorProvider.get()
                            .dotProductChunk(tmpKeyBatch, input, keyAttnWeights, 0, config.embeddingLength, chunkStart, chunkLength);
                    configurableTensorProvider.get()
                            .dotProductChunk(tmpValBatch, input, valueAttnWeights, 0, config.embeddingLength, chunkStart, chunkLength);
                }, splitSize);
            } else {
                qkvResults[0] = queryBatch;
                qkvResults[1] = tmpKeyBatch;
                qkvResults[2] = tmpValBatch;

                // compute the query vector
                VectorMath.pchunk(dctx.attentionSegmentStart, dctx.attentionSegmentLength, (chunkStart, chunkLength) -> {
                    configurableTensorProvider.get()
                            .dotProductBatchChunk(qkvResults, input, qkvWeights, 0, config.embeddingLength, chunkStart, chunkLength);
                }, splitSize);
            }

            queryAttnBias.ifPresent(
                    bias -> configurableTensorProvider.get().accumulate(queryBatch, bias, dctx.attentionSegmentStart, dctx.attentionSegmentLength)
            );
            keyAttnBias.ifPresent(
                    bias -> configurableTensorProvider.get().accumulate(tmpKeyBatch, bias, dctx.kvSegmentStart, dctx.kvSegmentLength)
            );
            valueAttnBias.ifPresent(
                    bias -> configurableTensorProvider.get().accumulate(tmpValBatch, bias, dctx.kvSegmentStart, dctx.kvSegmentLength)
            );

            debug("query", queryBatch, layerIndex);
            debug("key", tmpKeyBatch, layerIndex);
            debug("value", tmpValBatch, layerIndex);

            // This is our memory of the key and value vectors for each position
            for (int position = startPosition, bi = 0; position < startPosition + batchSize; position++, bi++) {
                int finalPosition = position;
                AbstractTensor key = kvMem.getKeyTensorForPosition(layerIndex, position);
                AbstractTensor val = kvMem.getValTensorForPosition(layerIndex, position);

                AbstractTensor[] kvp = kvMem.getKeyTensorsUptoPosition(layerIndex, position);
                AbstractTensor[] vvp = kvMem.getValTensorsUptoPosition(layerIndex, position);

                AbstractTensor tmpKey = tmpKeyBatch.slice(bi);
                AbstractTensor tmpVal = tmpValBatch.slice(bi);
                AbstractTensor query = queryBatch.slice(bi);
                AbstractTensor value = valueBatch.slice(bi);

                if (key.dType() != tmpKey.dType()) {
                    try (
                            AbstractTensor tmpKey2 = configurableTensorProvider.get().quantize(tmpKey, key.dType(), 0, config.kvLength);
                            AbstractTensor tmpVal2 = configurableTensorProvider.get().quantize(tmpVal, val.dType(), 0, config.kvLength)
                    ) {
                        key.copyFrom(
                                tmpKey2,
                                tmpKey2.getOffset(0, dctx.kvSegmentStart),
                                key.getOffset(0, dctx.kvSegmentStart),
                                dctx.kvSegmentLength
                        );
                        val.copyFrom(
                                tmpVal2,
                                tmpVal2.getOffset(0, dctx.kvSegmentStart),
                                val.getOffset(0, dctx.kvSegmentStart),
                                dctx.kvSegmentLength
                        );
                    }
                } else {
                    key.copyFrom(
                            tmpKey,
                            tmpKey.getOffset(0, dctx.kvSegmentStart),
                            key.getOffset(0, dctx.kvSegmentStart),
                            dctx.kvSegmentLength
                    );
                    val.copyFrom(
                            tmpVal,
                            tmpVal.getOffset(0, dctx.kvSegmentStart),
                            val.getOffset(0, dctx.kvSegmentStart),
                            dctx.kvSegmentLength
                    );
                }

                // apply RoPE if present (accounting for huggingface permutation)
                // https://github.com/huggingface/transformers/blob/d533465150532b0c5de167b574e59f64c68b1154/src/transformers/models/llama/convert_llama_weights_to_hf.py#L114
                config.ropeFreqs.ifPresent(rf -> {
                    int headPiece = config.headSize / 2;
                    int poffset = finalPosition * headPiece;

                    if (config.isGQA) {
                        // apply RoPE rotation to the q and k vectors for each head
                        for (int h = dctx.headStart; h < dctx.headEnd; h++) {
                            // get the q vectors for this head
                            int offset = h * config.headSize;

                            // skip if we are out of bounds
                            if (offset >= query.shape().last()) break;

                            int goffset = config.maybeMapToGroupHead(h) * config.headSize;
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

                        for (int h = dctx.groupHeadStart; h < dctx.groupHeadEnd; h++) {
                            // get the k vectors for this head
                            int offset = h * config.headSize;
                            if (offset >= key.shape().last()) break;
                            // rotate k by the freq theta and freq r
                            for (int i = offset; i < (offset + headPiece); i++) {
                                float k00 = key.get(0, i);
                                float k1 = key.get(0, i + headPiece); // hf permutation is 0,64,1,65 etc...
                                float[] f = rf[poffset + i];
                                float fcr = f[0];
                                float fci = f[1];
                                key.set(k00 * fcr - k1 * fci, 0, i);
                                key.set(k00 * fci + k1 * fcr, 0, i + headPiece);
                            }
                        }
                    } else {
                        // apply RoPE rotation to the q and k vectors for each head
                        for (int h = dctx.headStart; h < dctx.headEnd; h++) {
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

                // Attention
                VectorMath.pfor(dctx.headStart, dctx.headEnd, h -> {
                    int xoffset = config.maybeMapToGroupHead(h) * config.headSize;
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

                        if (config.attnLogitSoftCapping != null) {
                            for (int i = 0; i < finalPosition + 1; i++) {
                                float v = attn.get(0, i);
                                v /= config.attnLogitSoftCapping;
                                v = (float) FastMath.tanh(v);
                                v *= config.attnLogitSoftCapping;
                                attn.set(v, 0, i);
                            }
                        }

                        // softmax the scores to get attention weights, from 0..pos inclusively
                        VectorTensorMathUtils.softMax(attn, 0, finalPosition + 1);

                        // apply adjusted attention weights to value vectors
                        // do this for each position since the pages are not contiguous
                        for (int i = 0; i < vvp.length; i++) {
                            int len = vvp[i].shape().first(); // batch size
                            int offset = i * len;
                            int size = i == vvp.length - 1 ? (finalPosition + 1) - offset : len;
                            configurableTensorProvider.get().saxpy(attn, vvp[i], value, xoffset, yoffset, config.headSize, offset, 0, size);
                        }
                    }
                });
            }

            debug("after_attention", valueBatch, layerIndex);


            // matmul the projection and sum into input
            // input += c_proj_weight @ ybuf + c_proj_bias
            AbstractTensor result = m.makeDenseTensor(batchSize, config.embeddingLength);
            try (AbstractTensor vq = m.maybeQuantize(valueBatch)) {
                VectorMath.pchunk(0, config.embeddingLength, (chunkStart, chunkSize) -> {
                    configurableTensorProvider.get()
                            .dotProductChunk(
                                    result,
                                    vq,
                                    outputProjectionWeights,
                                    dctx.attentionSegmentStart,
                                    dctx.attentionSegmentLength,
                                    chunkStart,
                                    chunkSize
                            );
                }, splitSize);
                tensorReducer.ifPresent(func -> func.accept(Collections.singletonList(result)));
                outputProjectionBias.ifPresent(bias -> configurableTensorProvider.get().accumulate(result, bias, 0, config.embeddingLength));
            }

            return result;
        }
    }
}
