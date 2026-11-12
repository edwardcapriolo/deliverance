package io.teknek.reckoning.guided;

import io.teknek.reckoning.model.ResponseContext;
import io.teknek.reckoning.tensor2.TensorRef;

/**
 * Adapts request-time generation constraints to Reckoning's TensorRef logits.
 *
 * <p>A processor is created once for a generation request and is reused for every sampled token in that request. The
 * sampler calls {@link #process(TensorRef, ResponseContext)} after model-level logit transforms and before token
 * selection. After the sampler chooses a token, it calls {@link #accept(int, ResponseContext)} so stateful processors can
 * advance their internal guide state.</p>
 *
 * <p>Implementations should mutate {@code logits} in place. To disallow a token, set that token's logit to
 * {@link Float#NEGATIVE_INFINITY}. Implementations should not perform sampling, decode output text for display, or append
 * to {@link ResponseContext}; those responsibilities belong to the generation loop.</p>
 *
 * <p>The {@code responseContext} passed to {@link #accept(int, ResponseContext)} does not yet contain {@code tokenId}; the
 * generation loop appends the returned sampler token after the sampler returns. Stateful implementations should use the
 * {@code tokenId} argument as the source of truth for guide advancement.</p>
 */
public interface LogitsProcessorRef {
    void process(TensorRef logits, ResponseContext responseContext);

    default void accept(int tokenId, ResponseContext responseContext) {
    }
}
