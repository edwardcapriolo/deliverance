package io.teknek.deliverance.embedding;

import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/** Ports relevant tests from sentence-transformers/tests/sentence_transformer/modules/test_pooling.py. */
class SentenceTransformersPoolingPortedTest {
    private static final int DIM = 2;
    private static final int BATCH = 2;
    private static final int SEQ = 4;
    private static final int[] ATTENTION_MASK = { 1, 1, 1, 0, 1, 1, 1, 1 };

    @Test
    void testPoolingMeanAndMeanSqrtLenTokens() {
        try (AbstractTensor tokenEmbeddings = tensor(new float[][] {
                { 1.0f }, { 3.0f }, { 5.0f }
        })) {
            int[] attentionMask = { 1, 1, 0 };

            float[][] sentenceEmbedding = SentenceTransformersPooling.pool(tokenEmbeddings, attentionMask, 1, 3,
                    SentenceTransformersPooling.Mode.MEAN,
                    SentenceTransformersPooling.Mode.MEAN_SQRT_LEN_TOKENS);

            assertEquals(1, sentenceEmbedding.length);
            assertEquals(2, sentenceEmbedding[0].length);
            assertArrayEquals(new float[] { 2.0f, (float) (4.0 / Math.sqrt(2.0)) }, sentenceEmbedding[0], 1.0e-6f);
        }
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
            float[][] expected = concatExpected(modes);
            assertMatrixEquals(expected, output, 1.0e-5f);
        }
    }

    @Test
    void testNormalizeForward() {
        float[] embedding = { 3.0f, 4.0f };

        SentenceTransformersPooling.normalize(embedding);

        assertArrayEquals(new float[] { 0.6f, 0.8f }, embedding, 1.0e-6f);
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
