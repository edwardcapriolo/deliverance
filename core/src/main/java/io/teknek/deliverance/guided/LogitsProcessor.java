package io.teknek.deliverance.guided;

import io.teknek.deliverance.model.ResponseContext;
import io.teknek.deliverance.tensor.AbstractTensor;

/**
 * Adapts request-time generation constraints to Deliverance's logits tensor.
 *
 * <p>A processor is created once for a generation request and is reused for every sampled token in that request. The
 * sampler calls {@link #process(AbstractTensor, ResponseContext)} after model-level logit transforms and before token
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
public interface LogitsProcessor {
    /**
     * Applies this processor's constraints to the current vocabulary logits.
     *
     * @param logits mutable logits tensor for the next-token distribution; index {@code i} is token id {@code i}
     * @param responseContext accumulated response state before the next token is selected
     */
    void process(AbstractTensor logits, ResponseContext responseContext);

    /**
     * Notifies the processor that the sampler selected {@code tokenId}.
     *
     * <p>Stateless processors can ignore this callback. Stateful processors, such as guide-backed structured-output
     * processors, should advance their internal state here.</p>
     *
     * @param tokenId token selected by the sampler
     * @param responseContext accumulated response state before {@code tokenId} is appended
     */
    default void accept(int tokenId, ResponseContext responseContext) {
    }
}
