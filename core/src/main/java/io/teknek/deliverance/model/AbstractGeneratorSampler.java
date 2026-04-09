package io.teknek.deliverance.model;

import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.LayerNorm;
import io.teknek.deliverance.math.VectorMath;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.VectorTensorMathUtils;
import net.jafama.FastMath;

import java.util.Optional;
import java.util.PriorityQueue;
import java.util.Random;

import static io.teknek.deliverance.CausualWhisperer.LOGGER;

public abstract class AbstractGeneratorSampler {
    protected final AbstractModel model;
    protected final GeneratorParameters parameters;
    protected final LayerNorm layerNorm;
    protected final AbstractTensor output;
    protected final AbstractTensor logits;
    protected final Random random;
    protected final float uniformSample;

    public AbstractGeneratorSampler(AbstractModel model, GeneratorParameters generatorParameters,
                                    AbstractTensor output, AbstractTensor logits, LayerNorm layerNorm, Random random,
                                    float uniformSample) {
        this.model = model;
        this.parameters = generatorParameters;
        this.layerNorm = layerNorm;
        this.output = output;
        this.logits = logits;
        this.random = random;
        this.uniformSample = uniformSample;
    }
    public abstract SamplerReturn sample();
}

class DeliveranceLegacySampler extends AbstractGeneratorSampler{

    public DeliveranceLegacySampler(AbstractModel model, GeneratorParameters generatorParameters, AbstractTensor output,
                                    AbstractTensor logits, LayerNorm layerNorm, Random random, float uniformSample) {
        super(model, generatorParameters, output, logits, layerNorm, random, uniformSample);
    }

    @Override
    public SamplerReturn sample() {
        boolean logProbs = this.parameters.logProbs.orElse(false);
        int topLogProbs = this.parameters.topLogProbs.orElse(0);
        float temperature = this.parameters.temperature.orElse(0f);
        float xtcProbability = this.parameters.xtcProbability.orElse(0f);
        float xtcThreshold = this.parameters.xtcThreshold.orElse(0f);

        try (AbstractTensor embedding = layerNorm.forward(output)) {
            VectorMath.pchunk(0, model.config.vocabularySize, (chunkStart, chunkSize) -> {
                model.configurableTensorProvider.get()
                        .dotProductChunk(logits, embedding, model.sampleOutput.getOutputLogitsWeights(), 0,
                                model.config.embeddingLength, chunkStart, chunkSize);
            }, model.configurableTensorProvider.get().parallelSplitSize(), model.getPool());

            if (model.config.logitMultiplier != null) {
                LOGGER.debug("scaling logits logitMultiplier: {}", model.config.logitMultiplier);
                model.configurableTensorProvider.get().scale(1.0f / model.config.logitMultiplier,
                        logits, 0, model.config.vocabularySize);
            }
            int maxi = Integer.MIN_VALUE;
            double maxv = Double.NEGATIVE_INFINITY;
            PriorityQueue<IndexValueToken> topNLogProbs = new PriorityQueue<>();
            for (int i = 0; i < model.config.vocabularySize; i++) {
                float v = logits.get(0, i);
                if (model.config.finalLogitSoftCapping != null) {
                    v /= model.config.finalLogitSoftCapping;
                    v = (float) FastMath.tanh(v);
                    v = v * model.config.finalLogitSoftCapping;
                    logits.set(v, 0, i);
                }
                if (logProbs) {
                    IndexValueToken token = new IndexValueToken(i, v, model.getTokenizer().decode(i));
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
                AbstractTensor logSum = model.getTensorCache().getDirty(logits.dType(), logits.shape());
                VectorTensorMathUtils.logSumExpTensor(logSum, logits);
                for (IndexValueToken token : topNLogProbs) {
                    token.logProb = logSum.get(0, token.index);
                }
            }
            Optional<IndexValueToken> chosen = Optional.empty();
            if (xtcThreshold != 0){
                ExcludeTopChoicePicker p = new ExcludeTopChoicePicker(model, logits, xtcThreshold, xtcProbability, random);
                chosen = p.process();
            }
            if (temperature == 0.0) {
                if (chosen.isPresent() && chosen.get().index != maxi) {
                    if (LOGGER.isDebugEnabled()) {
                        LOGGER.debug("xtc: {} maxi: {}", model.tokenizer.decode(chosen.get().index), maxi);
                    }
                    return logProbs ? new SamplerReturn(chosen.get().index, topNLogProbs) : new SamplerReturn(chosen.get().index);
                } else {
                    return logProbs ? new SamplerReturn(maxi, topNLogProbs) : new SamplerReturn(maxi);
                }
            }
            float sum = 0;
            for (int i = 0; i < model.config.vocabularySize; i++) {
                float v = (float) FastMath.exp((logits.get(0, i) - maxv) / temperature);
                sum += v;
                logits.set(v, 0, i);
            }
            float acc = 0;

            for (int i = 0; i < model.config.vocabularySize; i++) {
                float v = logits.get(0, i) / sum;
                acc += v;
                if (acc >= uniformSample) {
                    LOGGER.debug("accumulator {} >= uniformSample {} returning {}", acc, uniformSample, i);
                    return logProbs ? new SamplerReturn(i, topNLogProbs) : new SamplerReturn(i);
                }
            }
            if (LOGGER.isDebugEnabled()) {
                LOGGER.debug("Reached end returning {}", model.config.vocabularySize - 1);
            }
            return logProbs ? new SamplerReturn(model.config.vocabularySize - 1, topNLogProbs)
                    : new SamplerReturn(model.config.vocabularySize - 1);

        }
    }
}