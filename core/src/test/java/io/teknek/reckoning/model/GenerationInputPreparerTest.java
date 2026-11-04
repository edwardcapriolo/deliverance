package io.teknek.reckoning.model;

import io.teknek.reckoning.math.ActivationFunction;
import io.teknek.reckoning.safetensors.Config;
import io.teknek.reckoning.tensor.AbstractTensor;
import io.teknek.reckoning.tensor.TensorDisplayUtil;
import io.teknek.reckoning.tensor.TensorShape;
import io.teknek.reckoning.tensor.impl.FloatBufferTensor;
import io.teknek.reckoning.tensor2.TensorRef;
import io.teknek.reckoning.tensor2.TensorRefBackedTensor;
import io.teknek.reckoning.tensor2.Lighter;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class GenerationInputPreparerTest {

    @Test
    public void decoderOnlyInputIdsAreCopiedAndSlicedToNextSequenceLength() {
        GenerationStepInputs inputs = GenerationInputPreparer.prepareInputsForGeneration(
                decoderOnlyConfig(),
                new int[]{10, 11, 12, 13},
                2,
                null,
                new int[]{1, 1, 1, 1},
                new int[]{1, 0, 1, 0},
                null,
                false,
                new int[]{0, 1, 2, 3},
                new int[]{7, 7, 8, 8},
                new int[]{0, 0, 9, 9},
                GenerationInputPreparerTest::newTensor,
                new Lighter()
        );

        assertFalse(inputs.usesInputEmbeds());
        assertArrayEquals(new int[]{12, 13}, inputs.inputIds());
        assertArrayEquals(new int[]{1, 1}, inputs.attentionMask());
        assertNull(inputs.encoderAttentionMask());
        assertArrayEquals(new int[]{2, 3}, inputs.positionIds());
        assertArrayEquals(new int[]{8, 8}, inputs.tokenTypeIds());
        assertArrayEquals(new int[]{9, 9}, inputs.mmTokenTypeIds());
        assertEquals(1, inputs.batchSize());
        assertEquals(2, inputs.sequenceLength());
        assertEquals(GenerationInputKey.INPUT_IDS, inputs.names().inputIdsKey());
    }

    @Test
    public void decoderOnlyFirstIterationUsesInputEmbedsWhenProvided() {
        try (TensorRef embeds = embeddings(1, 4, 3)) {
            GenerationStepInputs inputs = GenerationInputPreparer.prepareInputsForGeneration(
                    decoderOnlyConfig(),
                    new int[]{10, 11, 12, 13},
                    2,
                    null,
                    null,
                    null,
                    embeds,
                    true,
                    null,
                    null,
                    null,
                    GenerationInputPreparerTest::newTensor,
                    new Lighter()
            );

            try (TensorRef prepared = inputs.inputsEmbeds()) {
                assertTrue(inputs.usesInputEmbeds());
                assertNull(inputs.inputIds());
                assertEquals(1, inputs.batchSize());
                assertEquals(2, inputs.sequenceLength());
                assertEquals(normalizeDisplay("""
                        [0][0]=  6.0000 [0][1]=  7.0000 [0][2]=  8.0000 
                        [1][0]=  9.0000 [1][1]= 10.0000 [1][2]= 11.0000
                        """), normalizeDisplay(displayBatch(prepared, 0)));
            }
        }
    }

    @Test
    public void decoderOnlyFirstIterationUsesFullInputEmbedsWhenNextSequenceLengthIsNull() {
        try (TensorRef embeds = embeddings(1, 3, 2)) {
            GenerationStepInputs inputs = GenerationInputPreparer.prepareInputsForGeneration(
                    decoderOnlyConfig(),
                    new int[]{10, 11, 12},
                    null,
                    null,
                    null,
                    null,
                    embeds,
                    true,
                    null,
                    null,
                    null,
                    GenerationInputPreparerTest::newTensor,
                    new Lighter()
            );

            try (TensorRef prepared = inputs.inputsEmbeds()) {
                assertTrue(inputs.usesInputEmbeds());
                assertNull(inputs.inputIds());
                assertEquals(3, inputs.sequenceLength());
                assertEquals(normalizeDisplay("""
                        [0][0]=  0.0000 [0][1]=  1.0000 
                        [1][0]=  2.0000 [1][1]=  3.0000 
                        [2][0]=  4.0000 [2][1]=  5.0000
                        """), normalizeDisplay(displayBatch(prepared, 0)));
            }
        }
    }

    @Test
    public void decoderOnlyAfterFirstIterationUsesInputIdsEvenWhenEmbedsExist() {
        try (TensorRef embeds = embeddings(1, 4, 3)) {
            GenerationStepInputs inputs = GenerationInputPreparer.prepareInputsForGeneration(
                    decoderOnlyConfig(),
                    new int[]{10, 11, 12, 13},
                    1,
                    null,
                    null,
                    null,
                    embeds,
                    false,
                    null,
                    null,
                    null,
                    GenerationInputPreparerTest::newTensor,
                    new Lighter()
            );

            assertFalse(inputs.usesInputEmbeds());
            assertArrayEquals(new int[]{13}, inputs.inputIds());
            assertEquals(1, inputs.sequenceLength());
        }
    }

    @Test
    public void decoderOnlyUsesFullInputIdsWhenNextSequenceLengthIsNull() {
        GenerationStepInputs inputs = GenerationInputPreparer.prepareInputsForGeneration(
                decoderOnlyConfig(),
                new int[]{10, 11, 12, 13},
                null,
                null,
                new int[]{1, 1, 1, 1},
                null,
                null,
                false,
                new int[]{0, 1, 2, 3},
                null,
                null,
                GenerationInputPreparerTest::newTensor,
                new Lighter()
        );

        assertArrayEquals(new int[]{10, 11, 12, 13}, inputs.inputIds());
        assertArrayEquals(new int[]{1, 1, 1, 1}, inputs.attentionMask());
        assertArrayEquals(new int[]{0, 1, 2, 3}, inputs.positionIds());
        assertEquals(4, inputs.sequenceLength());
    }

    @Test
    public void encoderDecoderUsesDecoderInputNamesAndInputIds() {
        GenerationStepInputs inputs = GenerationInputPreparer.prepareInputsForGeneration(
                encoderDecoderConfig(),
                new int[]{4, 5, 6},
                2,
                null,
                new int[]{1, 1, 1},
                new int[]{1, 0, 1, 0},
                null,
                true,
                new int[]{0, 1, 2},
                null,
                new int[]{3, 4, 5},
                GenerationInputPreparerTest::newTensor,
                new Lighter()
        );

        assertArrayEquals(new int[]{5, 6}, inputs.inputIds());
        assertEquals(GenerationInputKey.DECODER_INPUT_IDS, inputs.names().inputIdsKey());
        assertEquals(GenerationInputKey.DECODER_POSITION_IDS, inputs.names().positionIdsKey());
        assertEquals(GenerationInputKey.DECODER_ATTENTION_MASK, inputs.names().attentionMaskKey());
        assertArrayEquals(new int[]{1, 2}, inputs.positionIds());
        assertArrayEquals(new int[]{1, 0, 1, 0}, inputs.encoderAttentionMask());
        assertArrayEquals(new int[]{4, 5}, inputs.mmTokenTypeIds());
    }

    @Test
    public void encoderDecoderUsesInputIdsEvenOnFirstIterationWhenEmbedsExist() {
        try (TensorRef embeds = embeddings(1, 3, 2)) {
            GenerationStepInputs inputs = GenerationInputPreparer.prepareInputsForGeneration(
                    encoderDecoderConfig(),
                    new int[]{4, 5, 6},
                    1,
                    null,
                    null,
                    null,
                    embeds,
                    true,
                    null,
                    null,
                    null,
                    GenerationInputPreparerTest::newTensor,
                    new Lighter()
            );

            assertFalse(inputs.usesInputEmbeds());
            assertArrayEquals(new int[]{6}, inputs.inputIds());
            assertEquals(GenerationInputKey.DECODER_INPUT_IDS, inputs.names().inputIdsKey());
            assertEquals(1, inputs.sequenceLength());
        }
    }

    @Test
    public void inputIdsAreRequiredWhenEmbedsAreNotUsed() {
        assertThrows(IllegalArgumentException.class, () -> GenerationInputPreparer.prepareInputsForGeneration(
                decoderOnlyConfig(),
                null,
                null,
                null,
                null,
                null,
                null,
                false,
                null,
                null,
                null,
                    GenerationInputPreparerTest::newTensor,
                    new Lighter()
        ));
    }

    @Test
    public void alignedInputsShorterThanPreparedSequenceAreRejected() {
        assertThrows(IllegalArgumentException.class, () -> GenerationInputPreparer.prepareInputsForGeneration(
                decoderOnlyConfig(),
                new int[]{10, 11, 12},
                null,
                null,
                new int[]{1, 1},
                null,
                null,
                false,
                null,
                null,
                null,
                    GenerationInputPreparerTest::newTensor,
                    new Lighter()
        ));
    }

    @Test
    public void nextSequenceLengthLongerThanInputIsRejected() {
        assertThrows(IllegalArgumentException.class, () -> GenerationInputPreparer.prepareInputsForGeneration(
                decoderOnlyConfig(),
                new int[]{10, 11, 12},
                4,
                null,
                null,
                null,
                null,
                false,
                null,
                null,
                null,
                    GenerationInputPreparerTest::newTensor,
                    new Lighter()
        ));
    }

    private static TensorRef embeddings(int batchSize, int sequenceLength, int embeddingLength) {
        AbstractTensor tensor = new FloatBufferTensor(batchSize, sequenceLength, embeddingLength);
        int value = 0;
        for (int batch = 0; batch < batchSize; batch++) {
            for (int position = 0; position < sequenceLength; position++) {
                for (int embedding = 0; embedding < embeddingLength; embedding++) {
                    tensor.set(value++, batch, position, embedding);
                }
            }
        }
        return TensorRef.owned(tensor);
    }

    private static TensorRef newTensor(TensorShape shape) {
        return TensorRef.owned(new FloatBufferTensor(shape.shapeArray()));
    }

    private static String displayBatch(TensorRef tensor, int batchIndex) {
        try (AbstractTensor batch = new TensorRefBackedTensor(tensor.slice(batchIndex))) {
            return TensorDisplayUtil.pretty2dDisplayAll(batch);
        }
    }

    private static String normalizeDisplay(String display) {
        return display.strip().replaceAll("(?m) +$", "");
    }

    private static Config decoderOnlyConfig() {
        return new Config(32, 16, 32, 2, 1, 1,
                1e-6f, 64, 2, List.of(1),
                ActivationFunction.Type.GELU_PYTORCH_TANH, null, null);
    }

    private static Config encoderDecoderConfig() {
        return new Config(32, 16, 32, 2, 1, 1,
                1e-6f, 64, 2, List.of(1),
                ActivationFunction.Type.GELU_PYTORCH_TANH, null, null, true);
    }
}
