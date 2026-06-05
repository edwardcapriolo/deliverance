package io.teknek.deliverance.model;

import io.teknek.deliverance.tensor.AbstractTensor;

/**
 * Typed model inputs for one generation forward step.
 *
 * <p>Every supported execution input is represented as an explicit field with a clear type and ownership contract.</p>
 *
 * <p>{@code inputsEmbeds}, when present, has shape {@code [batch, sequence, embedding]}. Dimension 0 is request batch
 * index, dimension 1 is token position within that request, and dimension 2 is the dense embedding vector.</p>
 */
public record GenerationStepInputs(
        GenerationInputNames names,
        int[] inputIds,
        Integer nextSequenceLength,
        PastKeyValues pastKeyValues,
        int[] attentionMask,
        int[] encoderAttentionMask,
        int[] positionIds,
        int[] tokenTypeIds,
        int[] mmTokenTypeIds,
        AbstractTensor inputsEmbeds,
        boolean firstIteration,
        int batchSize,
        int sequenceLength
) {

    public boolean usesInputEmbeds() {
        return inputsEmbeds != null;
    }
}
