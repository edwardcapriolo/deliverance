package io.teknek.deliverance.model;

import com.codahale.metrics.Histogram;
import io.teknek.deliverance.CausualWhisperer;
import io.teknek.deliverance.generator.LayerNorm;
import io.teknek.deliverance.math.VectorMath;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.VectorTensorMathUtils;
import net.jafama.FastMath;

import java.util.Optional;
import java.util.PriorityQueue;

public class GeneratorSampler {
    private final AbstractModel abstractModel;
    private final Histogram forward1;
    private final Histogram dotprod2;
    private final Histogram fullSample;

    private final AbstractTensor output;
    private final float temperature;
    private final float uniformSample;
    private final AbstractTensor logits;
    private final LayerNorm layerNorm;
    private final boolean logProbs;
    private final int topLogProbs;


    /**
     *
     * @param abstractModel
     * @param output
     * @param temperature
     * @param uniformSample
     * @param logits
     * @param layerNorm
     * @param logProbs if true calculate the log probs
     * @param topLogProbs The number of logprobs to calculate
     */
    public GeneratorSampler(AbstractModel abstractModel, AbstractTensor output, float temperature, float uniformSample,
                            AbstractTensor logits, LayerNorm layerNorm, boolean logProbs, int topLogProbs) {
        this.abstractModel = abstractModel;
        this.output = output;
        this.temperature = temperature;
        this.uniformSample = uniformSample;
        this.logits = logits;
        this.layerNorm = layerNorm;
        this.logProbs = logProbs;
        this.topLogProbs = topLogProbs;

        forward1 = abstractModel.metricRegistry.histogram("sample.foward1");
        dotprod2 = abstractModel.metricRegistry.histogram("sample.dotproduct2");
        fullSample = abstractModel.metricRegistry.histogram("sample.fullsample");
    }

    public SamplerReturn sample() {
        long start = System.nanoTime();
        try (AbstractTensor embedding = layerNorm.forward(output)) {
            long afterForward = System.nanoTime();
            forward1.update(Math.abs(afterForward - start));

            VectorMath.pchunk(0, abstractModel.config.vocabularySize, (chunkStart, chunkSize) -> {
                abstractModel.configurableTensorProvider.get()
                        .dotProductChunk(logits, embedding, abstractModel.sampleOutput.getOutputLogitsWeights(), 0,
                                abstractModel.config.embeddingLength, chunkStart, chunkSize);
            }, abstractModel.configurableTensorProvider.get().parallelSplitSize(), abstractModel.getPool());
            long afterDotProductChunk = System.nanoTime();
            dotprod2.update(Math.abs(afterDotProductChunk - afterForward));

            if (abstractModel.config.logitMultiplier != null) {
                CausualWhisperer.LOGGER.debug("scaling logits logitMultiplier: {}", abstractModel.config.logitMultiplier);
                abstractModel.configurableTensorProvider.get().scale(1.0f / abstractModel.config.logitMultiplier,
                        logits, 0, abstractModel.config.vocabularySize);
            }
            int maxi = Integer.MIN_VALUE;
            double maxv = Double.NEGATIVE_INFINITY;

            PriorityQueue<IndexValueToken> topNLogProbs = new PriorityQueue<>();
            for (int i = 0; i < abstractModel.config.vocabularySize; i++) {
                float v = logits.get(0, i);
                if (abstractModel.config.finalLogitSoftCapping != null) {
                    v /= abstractModel.config.finalLogitSoftCapping;
                    v = (float) FastMath.tanh(v);
                    v = v * abstractModel.config.finalLogitSoftCapping;
                    logits.set(v, 0, i);
                }
                if (this.logProbs) {
                    IndexValueToken token = new IndexValueToken(i, v, abstractModel.getTokenizer().decode(i));
                    topNLogProbs.offer(token);
                    if (topNLogProbs.size() > topLogProbs) {
                        topNLogProbs.poll();
                    }
                }
                if (v > maxv) {
                    maxi = i;
                    maxv = v;
                }
            }
            if (logProbs) {
                AbstractTensor logSum = abstractModel.getTensorCache().getDirty(logits.dType(), logits.shape());
                VectorTensorMathUtils.logSumExpTensor(logSum, logits);
                for (IndexValueToken token : topNLogProbs) {
                    token.logProb = logSum.get(0, token.index);
                }
            }

            if (temperature == 0.0) {
                CausualWhisperer.LOGGER.debug("temperature at 0 returning maxi {}", maxi);
                return logProbs ? new SamplerReturn(maxi, topNLogProbs) : new SamplerReturn(maxi);
            }
            float sum = 0;
            for (int i = 0; i < abstractModel.config.vocabularySize; i++) {
                float v = (float) FastMath.exp((logits.get(0, i) - maxv) / temperature);
                sum += v;
                logits.set(v, 0, i);
            }
            float acc = 0;

            for (int i = 0; i < abstractModel.config.vocabularySize; i++) {
                float v = logits.get(0, i) / sum;
                acc += v;
                if (acc >= uniformSample) {
                    CausualWhisperer.LOGGER.debug("accumulator {} >= uniformSample {} returning {}", acc, uniformSample, i);
                    return logProbs ? new SamplerReturn(i, topNLogProbs) : new SamplerReturn(i);
                }
            }
            CausualWhisperer.LOGGER.debug("Reached end returning {}", abstractModel.config.vocabularySize - 1);
            //return new SamplerReturn(abstractModel.config.vocabularySize - 1);
            return logProbs ? new SamplerReturn(abstractModel.config.vocabularySize - 1, topNLogProbs)
                    : new SamplerReturn(abstractModel.config.vocabularySize - 1);
        } finally {
            long end = System.nanoTime();
            fullSample.update(Math.abs(end - start));
        }
    }
}

/*
 This was the original snip from inside AbstractModel
    public int sample(AbstractTensor output, float temperature, float uniformSample, AbstractTensor logits) {
        long start = System.nanoTime();
        try (AbstractTensor embedding = sampleOutput.getOutputLayerNorm().forward(output)) {
            long afterForward = System.nanoTime();
            metricRegistry.histogram("sample.foward1").update(Math.abs(afterForward - start));
            // This is a mix of argmax and sampling with softmax
            VectorMath.pchunk(0, config.vocabularySize, (chunkStart, chunkSize) -> {
                configurableTensorProvider.get()
                        .dotProductChunk(logits, embedding, sampleOutput.getOutputLogitsWeights(), 0,
                                config.embeddingLength, chunkStart, chunkSize);
            }, configurableTensorProvider.get().parallelSplitSize());
            long afterDotProductChunk = System.nanoTime();
            metricRegistry.histogram("sample.dotproduct2").update(Math.abs(afterDotProductChunk - afterForward));

            if (config.logitMultiplier != null) {
                CausualWhisperer.LOGGER.debug("scaling logits logitMultiplier: {}", config.logitMultiplier);
                configurableTensorProvider.get().scale(1.0f / config.logitMultiplier, logits, 0, config.vocabularySize);
            }
            int maxi = Integer.MIN_VALUE;
            double maxv = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < config.vocabularySize; i++) {
                float v = logits.get(0, i);
                if (config.finalLogitSoftCapping != null) {
                    v /= config.finalLogitSoftCapping;
                    v = (float) FastMath.tanh(v);
                    v = v * config.finalLogitSoftCapping;
                    logits.set(v, 0, i);
                }
                if (v > maxv) {
                    maxi = i;
                    maxv = v;
                }
            }
            if (temperature == 0.0) {
                CausualWhisperer.LOGGER.debug("temperature at 0 returning maxi {}", maxi);
                return maxi;
            }
            float sum = 0;
            for (int i = 0; i < config.vocabularySize; i++) {
                float v = (float) FastMath.exp((logits.get(0, i) - maxv) / temperature);
                sum += v;
                logits.set(v, 0, i);
            }
            float acc = 0;
            for (int i = 0; i < config.vocabularySize; i++) {
                float v = logits.get(0, i) / sum;
                acc += v;
                if (acc >= uniformSample) {
                    CausualWhisperer.LOGGER.debug("accumulator {} >= uniformSample {} returning {}", acc, uniformSample, i);
                    return i;
                }
            }
            CausualWhisperer.LOGGER.debug("Reached end returning {}", config.vocabularySize - 1);
            return config.vocabularySize - 1;
        }
    }
     */
