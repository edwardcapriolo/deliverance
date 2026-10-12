package io.teknek.deliverance.model;

import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator2.LayerNorm2;
import io.teknek.deliverance.guided.LogitsProcessorRef;
import io.teknek.deliverance.tensor2.TensorRef;
import io.teknek.deliverance.DType;
import net.jafama.FastMath;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.PriorityQueue;
import java.util.Random;

/** TensorRef-native equivalent of {@link DeliveranceSampler}. */
final class DeliveranceSamplerRef extends AbstractGeneratorSamplerRef {
    private final TensorRef outputWeights;

    DeliveranceSamplerRef(AbstractModel model, GeneratorParameters parameters, TensorRef output,
            TensorRef logits, LayerNorm2 layerNorm, TensorRef outputWeights, Random random,
            ResponseContext responseContext, TensorRef argMaxScratch, Optional<LogitsProcessorRef> processor) {
        super(model, parameters, output, logits, layerNorm, random, responseContext, argMaxScratch, processor);
        this.outputWeights = outputWeights;
    }

    @Override
    public SamplerReturn sample() {
        try (TensorRef embedding = layerNorm.forward(output)) {
            outputProjection(embedding);
            if (model.getConfig().logitMultiplier != null) scale(1.0f / model.getConfig().logitMultiplier);
            if (model.getConfig().finalLogitSoftCapping != null) {
                float cap = model.getConfig().finalLogitSoftCapping;
                for (int i = 0; i < logits.shape().last(); i++) {
                    float value = logits.get(0, i) / cap;
                    logits.set((float) FastMath.tanh(value) * cap, 0, i);
                }
            }
            logitsProcessor.ifPresent(p -> p.process(logits, responseContext));
            if (parameters.xtcThreshold.isEmpty() && parameters.topK.isEmpty()
                    && parameters.topP.isEmpty() && parameters.uniformTopP.isEmpty()) {
                parameters.topP = Optional.of(0.10f);
            }
            boolean logProbs = parameters.logProbs.orElse(false);
            int topLogProbs = parameters.topLogProbs.orElse(0);
            float temperature = parameters.temperature.orElse(0.0f);
            validateSamplingParameters(parameters, temperature);
            PriorityQueue<IndexValueToken> top = topLogProbs(logProbs, topLogProbs);
            int greedy = argMax();
            float threshold = parameters.xtcThreshold.orElse(0.0f);
            if (temperature == 0.0f) {
                if (parameters.uniformTopP.isPresent()) {
                    throw new IllegalArgumentException("uniformTopP requires temperature > 0");
                }
                Optional<IndexValueToken> xtc = threshold == 0.0f ? Optional.empty()
                        : new ExcludeTopChoicePickerRef(model, lighter, logits, threshold,
                                parameters.xtcProbability.orElse(0.0f), random).process();
                int token = xtc.filter(t -> t.index != greedy).map(t -> t.index).orElse(greedy);
                logitsProcessor.ifPresent(p -> p.accept(token, responseContext));
                return logProbs ? new SamplerReturn(token, top) : new SamplerReturn(token);
            }
            if (parameters.topK.isPresent() && parameters.xtcThreshold.isPresent()) {
                throw new IllegalArgumentException("can not enable topk and xtc");
            }
            int token;
            if (threshold != 0.0f) {
                try (TensorRef scaled = lighter.allocate(DType.F32, logits.shape())) {
                    for (int i = 0; i < logits.shape().last(); i++) {
                        scaled.set(logits.get(0, i) / temperature, 0, i);
                    }
                    Optional<IndexValueToken> xtc = new ExcludeTopChoicePickerRef(model, lighter, scaled, threshold,
                            parameters.xtcProbability.orElse(0.0f), random).process();
                    token = xtc.filter(t -> t.index != greedy).map(t -> t.index).orElse(greedy);
                }
            } else {
                token = sample(temperature, random.nextFloat());
            }
            logitsProcessor.ifPresent(p -> p.accept(token, responseContext));
            return logProbs ? new SamplerReturn(token, top) : new SamplerReturn(token);
        }
    }

    private PriorityQueue<IndexValueToken> topLogProbs(boolean enabled, int limit) {
        PriorityQueue<IndexValueToken> result = new PriorityQueue<>();
        if (!enabled || limit <= 0) return result;
        for (int i = 0; i < logits.shape().last(); i++) {
            IndexValueToken token = new IndexValueToken(i, logits.get(0, i), model.decodeToken(i));
            if (result.size() < limit) result.offer(token);
            else if (token.compareTo(result.peek()) > 0) { result.poll(); result.offer(token); }
        }
        return result;
    }

    private int sample(float temperature, float uniformSample) {
        validateSamplingParameters(parameters, temperature);
        List<IndexValueToken> candidates = new ArrayList<>();
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < logits.shape().last(); i++) {
            float value = logits.get(0, i) / temperature;
            candidates.add(new IndexValueToken(i, value, null));
            max = Math.max(max, value);
        }
        candidates.sort(Comparator.comparingDouble((IndexValueToken t) -> t.value).reversed());
        int candidateCount = candidates.size();
        int limit = parameters.topK.map(k -> DeliveranceSampler.topKCandidateCount(k, candidateCount))
                .orElse(candidateCount);
        candidates = new ArrayList<>(candidates.subList(0, Math.min(limit, candidates.size())));
        float maxValue = max;
        double total = candidates.stream().mapToDouble(t -> FastMath.exp(t.value - maxValue)).sum();
        if (parameters.topP.isPresent()) {
            double cumulative = 0;
            int end = 0;
            while (end < candidates.size()) {
                cumulative += FastMath.exp(candidates.get(end).value - maxValue) / total;
                end++;
                if (cumulative >= parameters.topP.get()) break;
            }
            candidates = new ArrayList<>(candidates.subList(0, end));
            total = candidates.stream().mapToDouble(t -> FastMath.exp(t.value - maxValue)).sum();
            if (parameters.uniformTopP.isPresent()) {
                int pick = Math.min((int) (uniformSample * candidates.size()), candidates.size() - 1);
                return candidates.get(pick).index;
            }
        }
        double pick = uniformSample * total;
        for (IndexValueToken token : candidates) {
            pick -= FastMath.exp(token.value - maxValue);
            if (pick <= 0) return token.index;
        }
        return candidates.getLast().index;
    }

    /** Validates TensorRef sampling parameters with the legacy sampler's control-flow rules. */
    static void validateSamplingParameters(GeneratorParameters parameters, float temperature) {
        if (temperature <= 0.0f) {
            if (parameters.uniformTopP.isPresent()) {
                throw new IllegalArgumentException("uniformTopP requires temperature > 0");
            }
            return;
        }
        if (parameters.topP.isPresent() && (parameters.topP.get() <= 0.0f || parameters.topP.get() > 1.0f)) {
            throw new IllegalArgumentException("topP must be in (0, 1]");
        }
        if (parameters.uniformTopP.isPresent()
                && (parameters.uniformTopP.get() <= 0.0f || parameters.uniformTopP.get() > 1.0f)) {
            throw new IllegalArgumentException("uniformTopP must be in (0, 1]");
        }
        if (parameters.uniformTopP.isPresent() && parameters.topP.isEmpty()) {
            throw new IllegalArgumentException("uniformTopP requires topP");
        }
        if (parameters.topK.isPresent() && parameters.xtcThreshold.isPresent()) {
            throw new IllegalArgumentException("can not enable topk and xtc");
        }
    }
}
