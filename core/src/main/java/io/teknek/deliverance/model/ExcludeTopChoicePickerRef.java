package io.teknek.deliverance.model;

import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;

import java.util.Optional;
import java.util.Random;

/** TensorRef-native XTC picker. It intentionally has no legacy tensor conversion path. */
final class ExcludeTopChoicePickerRef {
    private final AbstractModel model;
    private final Lighter lighter;
    private final TensorRef logits;
    private final float threshold;
    private final float probability;
    private final Random random;

    ExcludeTopChoicePickerRef(AbstractModel model, Lighter lighter, TensorRef logits, float threshold,
            float probability, Random random) {
        this.model = model;
        this.lighter = lighter;
        this.logits = logits;
        this.threshold = threshold;
        this.probability = probability;
        this.random = random;
    }

    Optional<IndexValueToken> process() {
        if (random.nextFloat() > probability) {
            return Optional.empty();
        }
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < logits.shape().last(); i++) max = Math.max(max, logits.get(0, i));
        double sum = 0;
        for (int i = 0; i < logits.shape().last(); i++) sum += Math.exp(logits.get(0, i) - max);
        IndexValueToken first = null;
        IndexValueToken last = null;
        for (int i = 0; i < logits.shape().last(); i++) {
            float logProb = (float) (logits.get(0, i) - max - Math.log(sum));
            if (Math.exp(logProb) <= threshold) continue;
            IndexValueToken token = new IndexValueToken(i, logits.get(0, i), model.decodeToken(i));
            token.logProb = logProb;
            if (first == null || token.compareTo(first) < 0) first = token;
            if (last == null || token.compareTo(last) > 0) last = token;
        }
        return first == null ? Optional.empty() : Optional.of(
                model.isSpecialToken(last.index) ? last : first);
    }
}
