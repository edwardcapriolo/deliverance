package io.teknek.deliverance.model;

import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator2.LayerNorm2;
import io.teknek.deliverance.guided.LogitsProcessorRef;
import io.teknek.deliverance.tensor2.TensorRef;
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
            boolean logProbs = parameters.logProbs.orElse(false);
            int topLogProbs = parameters.topLogProbs.orElse(0);
            float temperature = parameters.temperature.orElse(0.0f);
            PriorityQueue<IndexValueToken> top = topLogProbs(logProbs, topLogProbs);
            int greedy = argMax();
            Optional<IndexValueToken> xtc = Optional.empty();
            float threshold = parameters.xtcThreshold.orElse(0.0f);
            if (threshold != 0.0f) {
                xtc = new ExcludeTopChoicePickerRef(model, lighter, logits, threshold,
                        parameters.xtcProbability.orElse(0.0f), random).process();
            }
            int token = temperature == 0.0f ? xtc.filter(t -> t.index != greedy).map(t -> t.index).orElse(greedy)
                    : sample(temperature);
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

    private int sample(float temperature) {
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
        }
        double pick = random.nextFloat() * total;
        for (IndexValueToken token : candidates) {
            pick -= FastMath.exp(token.value - maxValue);
            if (pick <= 0) return token.index;
        }
        return candidates.getLast().index;
    }
}
