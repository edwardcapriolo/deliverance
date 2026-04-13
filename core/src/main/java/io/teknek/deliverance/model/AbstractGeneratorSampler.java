package io.teknek.deliverance.model;

import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.LayerNorm;
import io.teknek.deliverance.math.VectorMath;
import io.teknek.deliverance.tensor.*;
import io.teknek.deliverance.tensorlib.ReadOnlyTensorMap;
import io.teknek.deliverance.tensorlib.Reduce;
import io.teknek.deliverance.tensorlib.TensorLib;

import io.teknek.dysfx.Maybe;

import io.teknek.dysfx.Something;
import io.teknek.dysfx.exception.UnreachableException;
import net.jafama.FastMath;

import java.util.*;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

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

class DeliveranceLegacySampler extends AbstractGeneratorSampler {

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


class DeliveranceSampler extends AbstractGeneratorSampler {

    public DeliveranceSampler(AbstractModel model, GeneratorParameters generatorParameters, AbstractTensor output,
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
                computeLogProbs(logits, topNLogProbs);
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
            if (parameters.topK.isPresent() && parameters.xtcThreshold.isPresent()) {
                throw new IllegalArgumentException("can not enable topk and xtc");
            }
            if (parameters.topK.isPresent()) {
                float topK = parameters.topK.get();
                LOGGER.info("max maxv {} maxi {} decoded {}", maxi, maxv, model.tokenizer.decode(maxi));
                try (AbstractTensor scaledLogits = model.getTensorCache().getDirty(logits.dType(), logits.shape())) {
                    for (int i = 0; i < model.config.vocabularySize; i++) {
                        float v = logits.get(0, i) / temperature;
                        scaledLogits.set(v, 0, i);
                    }
                    tensorMrSoftMax(scaledLogits, 0, scaledLogits.size());
                    SortedMap<Float, List<Integer>> buck = VectorTensorMathUtils.valueBuckets(scaledLogits);
                    int rePick = (int) scaledLogits.size() - ((int) (topK * scaledLogits.size()));
                    SortedMap<Float, List<Integer>> bucketsHighFirst = buck.reversed();

                    //consider forcing dtype 32
                    try (AbstractTensor inProb = model.getTensorCache().getDirty(logits.dType(), TensorShape.of(rePick));
                    AbstractTensor inLogits = model.getTensorCache().getDirty(logits.dType(), TensorShape.of(rePick))) {
                        List<String> inToks = new ArrayList<>();
                        int topPick = 0;
                        Iterator<Map.Entry<Float, List<Integer>>> jj = bucketsHighFirst.entrySet().iterator();
                        while (jj.hasNext() && topPick < rePick) {
                            Map.Entry<Float, List<Integer>> zz = jj.next();
                            for (int i = 0; i < zz.getValue().size(); i++) {
                                if (topPick >= rePick) {
                                    break;
                                } else {
                                    inProb.set(zz.getKey(), 0, topPick);
                                    inLogits.set(zz.getValue().get(i), 0, topPick);
                                    //tokens can have funky space
                                    inToks.add(zz.getValue().get(i) + " " + model.getTokenizer().decode(zz.getValue().get(i)));
                                    topPick++;
                                }
                            }
                        }
                        topPick--;
                        LOGGER.debug("topk {} logit {} token {} prob {}", topK, inLogits.get(0, topPick),
                                model.tokenizer.decode((int) inLogits.get(0, topPick)), inProb.get(0, topPick));
                        //System.out.println(TensorDisplayUtil.pretty2dDisplayAll(inProb));
                        //System.out.println(TensorDisplayUtil.pretty2dDisplayAll(inLogits));
                        //System.out.println(inToks);
                        return logProbs ? new SamplerReturn((int) inLogits.get(0, topPick), topNLogProbs) :
                                new SamplerReturn((int) inLogits.get(0, topPick));
                    }
                }
            } else if (xtcThreshold != 0){
                //topk is similar to xtc not allowing a stack
                PriorityQueue<IndexValueToken> topXLogProbs = new PriorityQueue<>();
                try (AbstractTensor scaledLogits = model.getTensorCache().getDirty(logits.dType(), logits.shape())) {
                    for (int i = 0; i < model.config.vocabularySize; i++) {
                        float v = logits.get(0, i) / temperature;
                        scaledLogits.set(v, 0, i);

                        if (logProbs) {
                            IndexValueToken token = new IndexValueToken(i, v, model.getTokenizer().decode(i));
                            topXLogProbs.offer(token);
                            if (topXLogProbs.size() > topLogProbs) {
                                topXLogProbs.poll();
                            }
                        }
                    }
                    if (logProbs) {
                        computeLogProbs(scaledLogits, topXLogProbs);
                    }
                    ExcludeTopChoicePicker picker = new ExcludeTopChoicePicker(model, scaledLogits, xtcThreshold, xtcProbability, random);
                    chosen = picker.process();
                    if (chosen.isPresent() && chosen.get().index != maxi) {
                        if (LOGGER.isDebugEnabled()) {
                            LOGGER.debug("xtc: {} maxi: {}", model.tokenizer.decode(chosen.get().index), maxi);
                        }
                        return logProbs ? new SamplerReturn(chosen.get().index, topXLogProbs) : new SamplerReturn(chosen.get().index);
                    } else {
                        return logProbs ? new SamplerReturn(maxi, topXLogProbs) : new SamplerReturn(maxi);
                    }
                }
            } else {
                throw new UnreachableException("There should be no way to get here");
            }
        }

    }

    public void computeLogProbs(AbstractTensor scaledLogits, PriorityQueue<IndexValueToken> logPobs){
        try (AbstractTensor logSum = model.getTensorCache().getDirty(scaledLogits.dType(), scaledLogits.shape())) {
            VectorTensorMathUtils.logSumExpTensor(logSum, scaledLogits);
            for (IndexValueToken token : logPobs) {
                token.logProb = logSum.get(0, token.index);
            }
        }
    }


    /*

    public class TemperatureScaling {
    public static double[] applyTemperature(double[] logits, double temperature) {
        // 1. Scale logits
        double[] scaledLogits = new double[logits.length];
        for (int i = 0; i < logits.length; i++) {
            scaledLogits[i] = logits[i] / temperature;
        }

        // 2. Softmax calculation
        double[] expLogits = new double[scaledLogits.length];
        double sumExpLogits = 0.0;

        // Find max for numerical stability
        double maxLogit = Arrays.stream(scaledLogits).max().orElse(0.0);

        for (int i = 0; i < scaledLogits.length; i++) {
            expLogits[i] = Math.exp(scaledLogits[i] - maxLogit);
            sumExpLogits += expLogits[i];
        }

        // Normalize to get probabilities
        double[] probabilities = new double[expLogits.length];
        for (int i = 0; i < expLogits.length; i++) {
            probabilities[i] = expLogits[i] / sumExpLogits;
        }

        return probabilities;
    }
}*/

    public void tensorMrSoftMax(AbstractTensor logits, long offset, long length){

        class MaxReadOnlyTensor implements ReadOnlyTensorMap<Float> {

            @Override
            public Float map(ReadableTensor t1, long offset, long length) {
                float max = Float.NEGATIVE_INFINITY; //correct
                for (long i = offset; i < offset + length; i++) {
                    float it = t1.get(0, (int)i);
                    if (it > max) {
                        max = it;
                    }
                }
                return max;
            }
        };

        Reduce<Float, Float> myReduce = t -> {
            Optional<Float> x = t.stream().max(Float::compareTo);
            return x.map(Maybe::possibly).orElseGet(Maybe::nothing);
        };

        TensorLib tensorLib = new TensorLib(model.getPool());
        Maybe<Float> maybeMaxValue = tensorLib.unary(logits).readOnlyMapper(new MaxReadOnlyTensor())
                .prepare(0, logits.size(), model.configurableTensorProvider.get().parallelSplitSize())
                        .reduce(myReduce);

        float max_val;
        if (maybeMaxValue instanceof Something<Float> something) {
            max_val = something.get();
        } else {
            throw new RuntimeException(maybeMaxValue + " isnt possible here");
        }
        float sum = 0.0f;
        for (int i = (int) offset; i < length; i++) {
            logits.set((float) FastMath.exp(logits.get(0, i) - max_val), 0, i);
            sum += logits.get(0, i);
        }

        model.configurableTensorProvider.get().scale(((float) (1.0/sum)), logits, (int) offset, (int) length );

    }
}

