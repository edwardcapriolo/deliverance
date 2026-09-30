package io.teknek.deliverance.guided;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.model.ResponseContext;
import io.teknek.deliverance.tensor2.TensorRef;
import io.teknek.sketches.guide.Guide;

import java.util.LinkedHashSet;
import java.util.Set;

/** Applies guide masks without converting the logits to a legacy tensor. */
final class GuideLogitsProcessorRef implements LogitsProcessorRef {
    private final Guide guide;
    private final MetricRegistry metrics;

    GuideLogitsProcessorRef(Guide guide, MetricRegistry metrics) {
        this.guide = guide;
        this.metrics = metrics;
    }

    @Override
    public void process(TensorRef logits, ResponseContext responseContext) {
        Set<Integer> allowed = new LinkedHashSet<>(guide.getTokens());
        int masked = 0;
        for (int i = 0; i < logits.shape().last(); i++) {
            if (!allowed.contains(i)) {
                logits.set(Float.NEGATIVE_INFINITY, 0, i);
                masked++;
            }
        }
        metrics.histogram("guided.allowed_tokens").update(allowed.size());
        metrics.histogram("guided.masked_tokens").update(masked);
    }

    @Override
    public void accept(int tokenId, ResponseContext responseContext) {
        guide.advance(tokenId);
    }
}
