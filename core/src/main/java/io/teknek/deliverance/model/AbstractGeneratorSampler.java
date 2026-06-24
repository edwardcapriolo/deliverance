package io.teknek.deliverance.model;

import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.LayerNorm;
import com.codahale.metrics.Timer;
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
        try (Timer.Context ignored = InferenceProfiler.timer(model.getMetricRegistry(), "sampler.sample").time()) {
        boolean logProbs = this.parameters.logProbs.orElse(false);
        int topLogProbs = this.parameters.topLogProbs.orElse(0);
        float temperature = this.parameters.temperature.orElse(0f);
        float xtcProbability = this.parameters.xtcProbability.orElse(0f);
        float xtcThreshold = this.parameters.xtcThreshold.orElse(0f);

        try (AbstractTensor embedding = layerNorm.forward(output)) {
            if (InferenceProfiler.isEnabled()) {
                InferenceProfiler.counter(model.getMetricRegistry(),
                        "sampler.output_input_" + embedding.dType()).inc();
                InferenceProfiler.counter(model.getMetricRegistry(),
                        "sampler.output_weight_" + model.sampleOutput.getOutputLogitsWeights().dType()).inc();
            }
            try (Timer.Context ignoredOutput = InferenceProfiler.timer(model.getMetricRegistry(), "sampler.output_projection").time()) {
                logits.clear();
                VectorMath.pchunk(0, model.config.vocabularySize, (chunkStart, chunkSize) -> {
                model.configurableTensorProvider.get()
                        .dotProductChunk(logits, embedding, model.sampleOutput.getOutputLogitsWeights(), 0,
                                model.config.embeddingLength, chunkStart, chunkSize);
                }, model.configurableTensorProvider.get().parallelSplitSize(), model.getPool());
            }

            if (model.config.logitMultiplier != null) {
                LOGGER.debug("scaling logits logitMultiplier: {}", model.config.logitMultiplier);
                model.configurableTensorProvider.get().scale(1.0f / model.config.logitMultiplier,
                        logits, 0, model.config.vocabularySize);
            }
            int maxi = Integer.MIN_VALUE;
            double maxv = Double.NEGATIVE_INFINITY;
            PriorityQueue<IndexValueToken> topNLogProbs = new PriorityQueue<>();
            try (Timer.Context ignoredScan = InferenceProfiler.timer(model.getMetricRegistry(), "sampler.logit_scan").time()) {
                for (int i = 0; i < model.config.vocabularySize; i++) {
                    float v = logits.get(0, i);
                    if (model.config.finalLogitSoftCapping != null) {
                        v /= model.config.finalLogitSoftCapping;
                        v = (float) FastMath.tanh(v);
                        v = v * model.config.finalLogitSoftCapping;
                        logits.set(v, 0, i);
                    }
                    if (logProbs) {
                        IndexValueToken token = new IndexValueToken(i, v, model.decodeToken(i));
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
            }
            if (logProbs) {
                AbstractTensor logSum = model.getTensorAllocator().getDirty(logits.dType(), logits.shape());
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
                        LOGGER.debug("xtc: {} maxi: {}", model.decodeToken(chosen.get().index), maxi);
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
}


class DeliveranceSampler extends AbstractGeneratorSampler {

    public DeliveranceSampler(AbstractModel model, GeneratorParameters generatorParameters, AbstractTensor output,
                                    AbstractTensor logits, LayerNorm layerNorm, Random random, float uniformSample) {
        super(model, generatorParameters, output, logits, layerNorm, random, uniformSample);
    }

    @Override
    public SamplerReturn sample() {
        try (Timer.Context ignored = InferenceProfiler.timer(model.getMetricRegistry(), "sampler.sample").time()) {
        boolean logProbs = this.parameters.logProbs.orElse(false);
        int topLogProbs = this.parameters.topLogProbs.orElse(0);
        float temperature = this.parameters.temperature.orElse(0f);
        float xtcProbability = this.parameters.xtcProbability.orElse(0f);
        float xtcThreshold = this.parameters.xtcThreshold.orElse(0f);
        if (parameters.xtcThreshold.isEmpty() && parameters.topK.isEmpty()) {
            parameters.topP = Optional.of(0.10f);
        }

        try (AbstractTensor embedding = layerNorm.forward(output)) {
            if (InferenceProfiler.isEnabled()) {
                InferenceProfiler.counter(model.getMetricRegistry(),
                        "sampler.output_input_" + embedding.dType()).inc();
                InferenceProfiler.counter(model.getMetricRegistry(),
                        "sampler.output_weight_" + model.sampleOutput.getOutputLogitsWeights().dType()).inc();
            }
            try (Timer.Context ignoredOutput = InferenceProfiler.timer(model.getMetricRegistry(), "sampler.output_projection").time()) {
                logits.clear();
                VectorMath.pchunk(0, model.config.vocabularySize, (chunkStart, chunkSize) -> {
                model.configurableTensorProvider.get()
                        .dotProductChunk(logits, embedding, model.sampleOutput.getOutputLogitsWeights(), 0,
                                model.config.embeddingLength, chunkStart, chunkSize);
                }, model.configurableTensorProvider.get().parallelSplitSize(), model.getPool());
            }

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
                    IndexValueToken token = new IndexValueToken(i, v, model.decodeToken(i));
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
                        LOGGER.debug("xtc: {} maxi: {}", model.decodeToken(chosen.get().index), maxi);
                    }
                    return logProbs ? new SamplerReturn(chosen.get().index, topNLogProbs) : new SamplerReturn(chosen.get().index);
                } else {
                    return logProbs ? new SamplerReturn(maxi, topNLogProbs) : new SamplerReturn(maxi);
                }
            }
            if (parameters.topK.isPresent() && parameters.xtcThreshold.isPresent()) {
                throw new IllegalArgumentException("can not enable topk and xtc");
            }
            if (xtcThreshold != 0){
                //topk is similar to xtc not allowing a stack
                PriorityQueue<IndexValueToken> topXLogProbs = new PriorityQueue<>();
                try (AbstractTensor scaledLogits = model.getTensorAllocator().getDirty(logits.dType(), logits.shape())) {
                    for (int i = 0; i < model.config.vocabularySize; i++) {
                        float v = logits.get(0, i) / temperature;
                        scaledLogits.set(v, 0, i);

                        if (logProbs) {
                            IndexValueToken token = new IndexValueToken(i, v, model.decodeToken(i));
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
                            LOGGER.debug("xtc: {} maxi: {}", model.decodeToken(chosen.get().index), maxi);
                        }
                        return logProbs ? new SamplerReturn(chosen.get().index, topXLogProbs) : new SamplerReturn(chosen.get().index);
                    } else {
                        return logProbs ? new SamplerReturn(maxi, topXLogProbs) : new SamplerReturn(maxi);
                    }
                }
            } else {
                int chosenToken;
                try (Timer.Context ignoredTopKTopP = InferenceProfiler.timer(model.getMetricRegistry(), "sampler.topk_topp").time()) {
                    chosenToken = sampleTopKTopP(logits, temperature, parameters.topK, parameters.topP, random.nextFloat());
                }
                return logProbs ? new SamplerReturn(chosenToken, topNLogProbs) : new SamplerReturn(chosenToken);
            }
        }

    }

    }

    class TopPSummary{
        double sum = 0.0;
        int count = 0;
        List<Integer> underCutoffLogits = new ArrayList<>();
        List<Float> underCutoffProb = new ArrayList<>();
        public TopPSummary ( Iterator<Map.Entry<Float, List<Integer>>> jj, float topP) {
            while (jj.hasNext() && sum < topP) {
                Map.Entry<Float, List<Integer>> zz = jj.next();
                for (int i = 0; i < zz.getValue().size(); i++) {
                    underCutoffLogits.add(zz.getValue().get(i));
                    underCutoffProb.add(zz.getKey());
                    sum = sum + zz.getKey();
                    count++;
                    if (sum >= topP) {
                        break;
                    }
                }
            }
        }
    }

    public void computeLogProbs(AbstractTensor scaledLogits, PriorityQueue<IndexValueToken> logPobs){
        try (AbstractTensor logSum = model.getTensorAllocator().getDirty(scaledLogits.dType(), scaledLogits.shape())) {
            VectorTensorMathUtils.logSumExpTensor(logSum, scaledLogits);
            for (IndexValueToken token : logPobs) {
                token.logProb = logSum.get(0, token.index);
            }
        }
    }

    static int topKCandidateCount(float topK, int vocabularySize) {
        if (topK <= 0) {
            throw new IllegalArgumentException("topK must be > 0");
        }
        if (topK < 1.0f) {
            return Math.max(1, vocabularySize - (int) (topK * vocabularySize));
        }
        return Math.min(vocabularySize, Math.round(topK));
    }

    static int sampleTopKTopP(AbstractTensor logits, float temperature, Optional<Float> topK, Optional<Float> topP,
            float uniformSample) {
        if (temperature <= 0.0f) {
            throw new IllegalArgumentException("temperature must be > 0 for sampling");
        }
        if (topP.isPresent() && (topP.get() <= 0.0f || topP.get() > 1.0f)) {
            throw new IllegalArgumentException("topP must be in (0, 1]");
        }
        int vocabularySize = Math.toIntExact(logits.size());
        int candidateLimit = topK.map(k -> topKCandidateCount(k, vocabularySize)).orElse(vocabularySize);
        List<IndexValueToken> candidates = new ArrayList<>(vocabularySize);
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < vocabularySize; i++) {
            float value = logits.get(0, i) / temperature;
            candidates.add(new IndexValueToken(i, value, null));
            if (value > max) {
                max = value;
            }
        }
        candidates.sort(Comparator.comparingDouble((IndexValueToken token) -> token.value).reversed());
        candidateLimit = Math.min(candidateLimit, candidates.size());
        List<IndexValueToken> filtered = new ArrayList<>(candidateLimit);
        for (int i = 0; i < candidateLimit; i++) {
            filtered.add(candidates.get(i));
        }
        double total = probabilitySum(filtered, max);
        if (topP.isPresent()) {
            double normalized = 0.0d;
            List<IndexValueToken> nucleus = new ArrayList<>();
            for (IndexValueToken token : filtered) {
                nucleus.add(token);
                normalized += FastMath.exp(token.value - max) / total;
                if (normalized >= topP.get()) {
                    break;
                }
            }
            filtered = nucleus;
            total = probabilitySum(filtered, max);
        }
        double pick = uniformSample * total;
        for (IndexValueToken token : filtered) {
            pick -= FastMath.exp(token.value - max);
            if (pick <= 0.0d) {
                return token.index;
            }
        }
        return filtered.getLast().index;
    }

    private static double probabilitySum(List<IndexValueToken> tokens, float max) {
        double total = 0.0d;
        for (IndexValueToken token : tokens) {
            total += FastMath.exp(token.value - max);
        }
        return total;
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
