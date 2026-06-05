package io.teknek.deliverance.model;

import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorShape;

import java.util.Arrays;
import java.util.function.Function;

final class GenerationInputPreparer {
    private GenerationInputPreparer() {
    }

    static GenerationStepInputs prepareInputsForGeneration(
            Config config,
            int[] inputIds,
            Integer nextSequenceLength,
            PastKeyValues pastKeyValues,
            int[] attentionMask,
            int[] encoderAttentionMask,
            AbstractTensor inputsEmbeds,
            boolean firstIteration,
            int[] positionIds,
            int[] tokenTypeIds,
            int[] mmTokenTypeIds,
            Function<TensorShape, AbstractTensor> tensorFactory
    ) {
        GenerationInputNames names = GenerationInputNames.forConfig(config);
        int[] preparedInputIds = null;
        AbstractTensor preparedInputsEmbeds = null;
        int batchSize;
        int sequenceLength;

        if (!config.isEncoderDecoder && inputsEmbeds != null && firstIteration) {
            preparedInputsEmbeds = copyLastEmbeddings(inputsEmbeds, nextSequenceLength, tensorFactory);
            batchSize = preparedInputsEmbeds.shape().first();
            sequenceLength = preparedInputsEmbeds.shape().dim(1);
        } else {
            preparedInputIds = copyLast(inputIds, nextSequenceLength, "inputIds");
            batchSize = 1;
            sequenceLength = preparedInputIds.length;
        }

        int[] preparedAttentionMask = copyMatchingLength(attentionMask, sequenceLength, "attentionMask");
        int[] preparedEncoderAttentionMask = config.isEncoderDecoder && encoderAttentionMask != null
                ? Arrays.copyOf(encoderAttentionMask, encoderAttentionMask.length)
                : null;
        int[] preparedPositionIds = copyMatchingLength(positionIds, sequenceLength, names.positionIdsKey().externalName());
        int[] preparedTokenTypeIds = copyMatchingLength(tokenTypeIds, sequenceLength, "token_type_ids");
        int[] preparedMmTokenTypeIds = copyMatchingLength(mmTokenTypeIds, sequenceLength, "mm_token_type_ids");

        return new GenerationStepInputs(
                names,
                preparedInputIds,
                nextSequenceLength,
                pastKeyValues,
                preparedAttentionMask,
                preparedEncoderAttentionMask,
                preparedPositionIds,
                preparedTokenTypeIds,
                preparedMmTokenTypeIds,
                preparedInputsEmbeds,
                firstIteration,
                batchSize,
                sequenceLength
        );
    }

    private static int[] copyLast(int[] values, Integer nextSequenceLength, String fieldName) {
        if (values == null) {
            throw new IllegalArgumentException(fieldName + " must not be null when inputsEmbeds are not used");
        }
        int length = selectedLength(values.length, nextSequenceLength, fieldName);
        return Arrays.copyOfRange(values, values.length - length, values.length);
    }

    private static int[] copyMatchingLength(int[] values, int sequenceLength, String fieldName) {
        if (values == null) {
            return null;
        }
        if (values.length == sequenceLength) {
            return Arrays.copyOf(values, values.length);
        }
        if (values.length < sequenceLength) {
            throw new IllegalArgumentException(fieldName + " length " + values.length
                    + " is shorter than sequenceLength " + sequenceLength);
        }
        return Arrays.copyOfRange(values, values.length - sequenceLength, values.length);
    }

    private static AbstractTensor copyLastEmbeddings(
            AbstractTensor inputsEmbeds,
            Integer nextSequenceLength,
            Function<TensorShape, AbstractTensor> tensorFactory
    ) {
        if (inputsEmbeds.shape().dims() != 3) {
            throw new IllegalArgumentException("inputsEmbeds must have shape [batch, sequence, embedding]");
        }
        int batchSize = inputsEmbeds.shape().first();
        int sourceSequenceLength = inputsEmbeds.shape().dim(1);
        int embeddingLength = inputsEmbeds.shape().last();
        int selectedSequenceLength = selectedLength(sourceSequenceLength, nextSequenceLength, "inputsEmbeds");
        int sourceStart = sourceSequenceLength - selectedSequenceLength;
        AbstractTensor copy = tensorFactory.apply(TensorShape.of(batchSize, selectedSequenceLength, embeddingLength));
        for (int batch = 0; batch < batchSize; batch++) {
            copy.copyFrom(
                    inputsEmbeds,
                    inputsEmbeds.getOffset(batch, sourceStart, 0),
                    copy.getOffset(batch, 0, 0),
                    selectedSequenceLength * embeddingLength
            );
        }
        return copy;
    }

    private static int selectedLength(int sourceLength, Integer nextSequenceLength, String fieldName) {
        if (nextSequenceLength == null) {
            return sourceLength;
        }
        if (nextSequenceLength < 0) {
            throw new IllegalArgumentException("nextSequenceLength must not be negative");
        }
        if (nextSequenceLength > sourceLength) {
            throw new IllegalArgumentException("nextSequenceLength " + nextSequenceLength
                    + " is longer than " + fieldName + " length " + sourceLength);
        }
        return nextSequenceLength;
    }
}
