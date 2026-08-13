package io.teknek.deliverance.sentence_transformer.modules;

import io.teknek.deliverance.embedding.SentenceTransformersPooling;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/** Ports sentence-transformers/tests/sentence_transformer/modules/test_pooling.py. */
class TestPoolingPortedTest {
    private static final int DIM = 2;
    private static final int BATCH = 2;
    private static final int SEQ = 4;
    private static final int[] ATTENTION_MASK = { 1, 1, 1, 0, 1, 1, 1, 1 };

    @Disabled("Requires prompt handling on SentenceTransformer.encode output dictionaries.")
    @Test
    void testPoolingPromptAttentionMaskRespectsIncludePrompt() {
    }

    @ParameterizedTest
    @EnumSource(SentenceTransformersPooling.Mode.class)
    void testPoolingForwardAllStrategies(SentenceTransformersPooling.Mode poolingMode) {
        try (AbstractTensor tokenEmbeddings = new FloatBufferTensor(3 * 5, 8)) {
            int[] attentionMask = {
                    1, 1, 1, 0, 0,
                    0, 1, 1, 1, 1,
                    1, 1, 1, 1, 1
            };
            float[][] sentenceEmbedding = SentenceTransformersPooling.pool(tokenEmbeddings, attentionMask, 3, 5,
                    poolingMode);

            assertEquals(3, sentenceEmbedding.length);
            assertEquals(8, sentenceEmbedding[0].length);
        }
    }

    @Disabled("Gradient flow is a PyTorch autograd concern and has no Java equivalent.")
    @Test
    void testPoolingGradientFlow() {
    }

    @Test
    @Disabled("Deliverance pooling helper does not expose a separate cls_token_embeddings feature yet.")
    void testPoolingClsUsesClsTokenEmbeddings() {
    }

    @Test
    void testPoolingClsRightPaddedUsesPositionZero() {
        try (AbstractTensor tokenEmbeddings = tensor(new float[][] {
                { 1.0f }, { 2.0f }, { 3.0f }, { 4.0f },
                { 5.0f }, { 6.0f }, { 7.0f }, { 8.0f }
        })) {
            float[][] output = SentenceTransformersPooling.pool(tokenEmbeddings,
                    new int[] { 1, 1, 1, 0, 1, 1, 0, 0 }, 2, 4,
                    SentenceTransformersPooling.Mode.CLS);

            assertArrayEquals(new float[] { 1.0f }, output[0]);
            assertArrayEquals(new float[] { 5.0f }, output[1]);
        }
    }

    @Test
    void testPoolingClsLeftPaddedFindsFirstRealToken() {
        try (AbstractTensor tokenEmbeddings = tensor(new float[][] {
                { 1.0f }, { 2.0f }, { 3.0f }, { 4.0f },
                { 5.0f }, { 6.0f }, { 7.0f }, { 8.0f }
        })) {
            float[][] output = SentenceTransformersPooling.pool(tokenEmbeddings,
                    new int[] { 0, 0, 1, 1, 0, 1, 1, 1 }, 2, 4,
                    SentenceTransformersPooling.Mode.CLS);

            assertArrayEquals(new float[] { 3.0f }, output[0]);
            assertArrayEquals(new float[] { 6.0f }, output[1]);
        }
    }

    @Test
    void testPoolingMaxRespectsAttentionMask() {
        try (AbstractTensor tokenEmbeddings = tensor(new float[][] { { 1.0f }, { 3.0f }, { 5.0f }, { 10.0f } })) {
            float[][] output = SentenceTransformersPooling.pool(tokenEmbeddings,
                    new int[] { 1, 1, 1, 0 }, 1, 4, SentenceTransformersPooling.Mode.MAX);

            assertArrayEquals(new float[] { 5.0f }, output[0]);
        }
    }

    @Test
    void testPoolingMeanAndMeanSqrtLenTokens() {
        try (AbstractTensor tokenEmbeddings = tensor(new float[][] { { 1.0f }, { 3.0f }, { 5.0f } })) {
            float[][] output = SentenceTransformersPooling.pool(tokenEmbeddings, new int[] { 1, 1, 0 }, 1, 3,
                    SentenceTransformersPooling.Mode.MEAN,
                    SentenceTransformersPooling.Mode.MEAN_SQRT_LEN_TOKENS);

            assertArrayEquals(new float[] { 2.0f, (float) (4.0 / Math.sqrt(2.0)) }, output[0], 1.0e-6f);
        }
    }

    @Test
    void testPoolingWeightedmeanRespectsAttentionMask() {
        try (AbstractTensor tokenEmbeddings = tensor(new float[][] { { 1.0f }, { 3.0f }, { 10.0f } })) {
            float[][] output = SentenceTransformersPooling.pool(tokenEmbeddings, new int[] { 1, 1, 0 }, 1, 3,
                    SentenceTransformersPooling.Mode.WEIGHTED_MEAN);

            assertArrayEquals(new float[] { 7.0f / 3.0f }, output[0], 1.0e-6f);
        }
    }

    @Test
    void testPoolingLasttokenFindsLastAttendedToken() {
        try (AbstractTensor tokenEmbeddings = tensor(new float[][] {
                { 0.0f }, { 1.0f }, { 2.0f }, { 3.0f },
                { 5.0f }, { 6.0f }, { 7.0f }, { 8.0f }
        })) {
            float[][] output = SentenceTransformersPooling.pool(tokenEmbeddings,
                    new int[] { 1, 1, 1, 0, 1, 1, 0, 0 }, 2, 4,
                    SentenceTransformersPooling.Mode.LAST_TOKEN);

            assertArrayEquals(new float[] { 2.0f }, output[0]);
            assertArrayEquals(new float[] { 6.0f }, output[1]);
        }
    }

    @Test
    void testPoolingLasttokenAllPaddingReturnsZeroVector() {
        try (AbstractTensor tokenEmbeddings = tensor(new float[][] { { 1.0f, 1.0f }, { 1.0f, 1.0f },
                { 1.0f, 1.0f }, { 1.0f, 1.0f } })) {
            float[][] output = SentenceTransformersPooling.pool(tokenEmbeddings, new int[] { 0, 0, 0, 0 }, 1, 4,
                    SentenceTransformersPooling.Mode.LAST_TOKEN);

            assertArrayEquals(new float[] { 0.0f, 0.0f }, output[0]);
        }
    }

    @Disabled("include_prompt=false is not wired into Deliverance embedding pooling yet.")
    @Test
    void testPoolingExcludesPromptTokensDirectly() {
    }

    @ParameterizedTest
    @EnumSource(SentenceTransformersPooling.Mode.class)
    void testPoolingExactValues(SentenceTransformersPooling.Mode poolingMode) {
        try (AbstractTensor tokenEmbeddings = tokenEmbeddings()) {
            float[][] output = SentenceTransformersPooling.pool(tokenEmbeddings, ATTENTION_MASK, BATCH, SEQ,
                    poolingMode);
            assertMatrixEquals(expectedByMode(poolingMode), output, 1.0e-5f);
        }
    }

    @ParameterizedTest
    @MethodSource("multiModes")
    void testPoolingMultiMode(SentenceTransformersPooling.Mode[] modes) {
        try (AbstractTensor tokenEmbeddings = tokenEmbeddings()) {
            float[][] output = SentenceTransformersPooling.pool(tokenEmbeddings, ATTENTION_MASK, BATCH, SEQ, modes);
            assertMatrixEquals(concatExpected(modes), output, 1.0e-5f);
        }
    }

    @Disabled("include_prompt=false is not wired into Deliverance embedding pooling yet.")
    @Test
    void testPoolingExcludesPromptTokensForPaddedAndFlattenedInputs() {
    }

    @Disabled("Legacy Python bool kwargs/config conversion are specific to SentenceTransformers module serialization.")
    @Test
    void testPoolingLegacyBoolKwargsWithDeprecationWarning() {
    }

    @Disabled("Legacy Python bool kwargs/config conversion are specific to SentenceTransformers module serialization.")
    @Test
    void testPoolingLegacyMultipleBoolKwargs() {
    }

    @Disabled("Legacy Python bool kwargs/config conversion are specific to SentenceTransformers module serialization.")
    @Test
    void testPoolingLegacyConfigConversion() {
    }

    @Disabled("Legacy Python bool kwargs/config conversion are specific to SentenceTransformers module serialization.")
    @Test
    void testPoolingLegacyConfigConversionMultiMode() {
    }

    @Disabled("Pooling module JSON save/load is not implemented as a separate Deliverance module.")
    @Test
    void testPoolingConfigRoundTrip() {
    }

    @Disabled("Invalid Python module constructor mode validation is not applicable to Deliverance enum modes.")
    @Test
    void testPoolingInvalidModeRaises() {
    }

    @Disabled("Requires CUDA/flash-attention-specific upstream comparison.")
    @Test
    void testPoolingFlattenedLiveFlashAttention() {
    }

    private static Stream<Arguments> multiModes() {
        return Stream.of(
                Arguments.of((Object) new SentenceTransformersPooling.Mode[] {
                        SentenceTransformersPooling.Mode.CLS, SentenceTransformersPooling.Mode.MEAN }),
                Arguments.of((Object) new SentenceTransformersPooling.Mode[] {
                        SentenceTransformersPooling.Mode.MEAN,
                        SentenceTransformersPooling.Mode.MEAN_SQRT_LEN_TOKENS }),
                Arguments.of((Object) new SentenceTransformersPooling.Mode[] {
                        SentenceTransformersPooling.Mode.CLS,
                        SentenceTransformersPooling.Mode.MAX,
                        SentenceTransformersPooling.Mode.MEAN,
                        SentenceTransformersPooling.Mode.MEAN_SQRT_LEN_TOKENS,
                        SentenceTransformersPooling.Mode.WEIGHTED_MEAN,
                        SentenceTransformersPooling.Mode.LAST_TOKEN }));
    }

    private static AbstractTensor tokenEmbeddings() {
        return tensor(new float[][] {
                { 1.0f, 2.0f }, { 3.0f, 4.0f }, { 5.0f, 6.0f }, { 99.0f, 99.0f },
                { 10.0f, 20.0f }, { 30.0f, 40.0f }, { 50.0f, 60.0f }, { 70.0f, 80.0f }
        });
    }

    private static float[][] expectedByMode(SentenceTransformersPooling.Mode mode) {
        return switch (mode) {
            case CLS -> new float[][] { { 1.0f, 2.0f }, { 10.0f, 20.0f } };
            case MAX -> new float[][] { { 5.0f, 6.0f }, { 70.0f, 80.0f } };
            case MEAN -> new float[][] { { 3.0f, 4.0f }, { 40.0f, 50.0f } };
            case MEAN_SQRT_LEN_TOKENS -> new float[][] {
                    { (float) (9.0 / Math.sqrt(3.0)), (float) (12.0 / Math.sqrt(3.0)) },
                    { 80.0f, 100.0f } };
            case WEIGHTED_MEAN -> new float[][] { { 22.0f / 6.0f, 28.0f / 6.0f }, { 50.0f, 60.0f } };
            case LAST_TOKEN -> new float[][] { { 5.0f, 6.0f }, { 70.0f, 80.0f } };
        };
    }

    private static float[][] concatExpected(SentenceTransformersPooling.Mode[] modes) {
        float[][] output = new float[BATCH][DIM * modes.length];
        for (int modeIndex = 0; modeIndex < modes.length; modeIndex++) {
            float[][] expected = expectedByMode(modes[modeIndex]);
            for (int batch = 0; batch < BATCH; batch++) {
                System.arraycopy(expected[batch], 0, output[batch], modeIndex * DIM, DIM);
            }
        }
        return output;
    }

    private static AbstractTensor tensor(float[][] values) {
        AbstractTensor tensor = new FloatBufferTensor(values.length, values[0].length);
        for (int row = 0; row < values.length; row++) {
            for (int col = 0; col < values[row].length; col++) {
                tensor.set(values[row][col], row, col);
            }
        }
        return tensor;
    }

    private static void assertMatrixEquals(float[][] expected, float[][] actual, float delta) {
        assertEquals(expected.length, actual.length);
        for (int row = 0; row < expected.length; row++) {
            assertArrayEquals(expected[row], actual[row], delta, "row=" + row);
        }
    }
}
