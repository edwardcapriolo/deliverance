package io.teknek.deliverance.model.diffusiongemma;

import io.teknek.deliverance.tensor.AbstractTensor;

/**
 * Mutates DiffusionGemma denoising logits in place for a specific denoising step.
 *
 * <p>This is the diffusion counterpart to autoregressive logits processors, but it intentionally does not reuse the AR
 * {@code guided.LogitsProcessor} interface. DiffusionGemma processors operate inside the canvas denoising loop, where the
 * relevant state is the current denoising step rather than the generated response text or last accepted token.</p>
 *
 * <p>Implementations are expected to be created once for a generation request and reused for each denoising step. They
 * should mutate {@code logits} in place and should not sample tokens, update the canvas, or advance stopping criteria;
 * those responsibilities belong to the DiffusionGemma generation loop and sampler.</p>
 */
public interface DiffusionGemmaLogitsProcessor {
    /**
     * Applies this processor to the mutable canvas logits for {@code curStep}.
     *
     * @param logits mutable logits tensor, normally shaped {@code [batch, canvasLength, vocabSize]}
     * @param curStep current denoising step in the countdown schedule; processor implementations define the valid range
     */
    void process(AbstractTensor logits, int curStep);
}
