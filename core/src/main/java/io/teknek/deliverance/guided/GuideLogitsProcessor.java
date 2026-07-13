package io.teknek.deliverance.guided;

import com.codahale.metrics.MetricRegistry;
import com.codahale.metrics.Timer;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.model.ResponseContext;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.sketches.guide.Guide;

import java.util.LinkedHashSet;
import java.util.Set;

final class GuideLogitsProcessor implements LogitsProcessor {
    private final Guide guide;
    private final MetricRegistry metricRegistry;

    GuideLogitsProcessor(Guide guide, MetricRegistry metricRegistry) {
        this.guide = guide;
        this.metricRegistry = metricRegistry;
    }

    @Override
    public void process(AbstractTensor logits, ResponseContext responseContext) {
        try (Timer.Context ignored = InferenceProfiler.timer(metricRegistry, "guided.logits_process").time()) {
            Set<Integer> allowedTokens = new LinkedHashSet<>(guide.getTokens());
            metricRegistry.histogram("guided.allowed_tokens").update(allowedTokens.size());
            int masked = 0;
            for (int i = 0; i < logits.size(); i++) {
                if (!allowedTokens.contains(i)) {
                    logits.set(Float.NEGATIVE_INFINITY, 0, i);
                    masked++;
                }
            }
            metricRegistry.histogram("guided.masked_tokens").update(masked);
            if (InferenceProfiler.isEnabled()) {
                InferenceProfiler.counter(metricRegistry, "guided.allowed_tokens.total").inc(allowedTokens.size());
                InferenceProfiler.counter(metricRegistry, "guided.masked_tokens.total").inc(masked);
            }
        }
    }

    @Override
    public void accept(int tokenId, ResponseContext responseContext) {
        try (Timer.Context ignored = InferenceProfiler.timer(metricRegistry, "guided.accept").time()) {
            guide.advance(tokenId);
        }
    }
}
