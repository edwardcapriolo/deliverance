package io.teknek.deliverance.guided;

import io.teknek.deliverance.model.ResponseContext;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.sketches.guide.Guide;

import java.util.LinkedHashSet;
import java.util.Set;

final class GuideLogitsProcessor implements LogitsProcessor {
    private final Guide guide;

    GuideLogitsProcessor(Guide guide) {
        this.guide = guide;
    }

    @Override
    public void process(AbstractTensor logits, ResponseContext responseContext) {
        Set<Integer> allowedTokens = new LinkedHashSet<>(guide.getTokens());
        for (int i = 0; i < logits.size(); i++) {
            if (!allowedTokens.contains(i)) {
                logits.set(Float.NEGATIVE_INFINITY, 0, i);
            }
        }
    }

    @Override
    public void accept(int tokenId, ResponseContext responseContext) {
        guide.advance(tokenId);
    }
}
