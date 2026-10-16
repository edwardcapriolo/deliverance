package io.teknek.deliverance.math;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Unit tests for the {@code VectorMathUtils} class, which provides utility methods for
 * vectorized mathematical operations. These tests cover various scenarios, including computing
 * outer products, rotary frequencies, similarity, and in-place normalization behavior.
 */
class VectorMathUtilsTest {

    @Test
    void computesOuterProductInRowMajorOrder() {
        assertArrayEquals(new float[]{3.0f, 4.0f, 6.0f, 8.0f},
                VectorMathUtils.outerProduct(new float[]{1.0f, 2.0f}, new float[]{3.0f, 4.0f}));
    }

    @Test
    void computesOuterProductWithAnEmptyOperand() {
        assertArrayEquals(new float[0],
                VectorMathUtils.outerProduct(new float[]{1.0f, 2.0f}, new float[0]));
    }

    @Test
    void precomputesRotaryFrequenciesForEachPositionAndFrequency() {
        float[][] frequencies = VectorMathUtils.precomputeFreqsCis(4, 3, 10_000.0, 1.0);

        assertEquals(6, frequencies.length);
        assertArrayEquals(new float[]{1.0f, 0.0f}, frequencies[0], 1.0e-6f);
        assertArrayEquals(new float[]{(float) Math.cos(1.0), (float) Math.sin(1.0)},
                frequencies[2], 1.0e-6f);
        assertArrayEquals(new float[]{(float) Math.cos(0.02), (float) Math.sin(0.02)},
                frequencies[5], 1.0e-6f);
    }

    @Test
    void computesCosineSimilarity() {
        assertEquals(1.0f,
                VectorMathUtils.cosineSimilarity(new float[]{3.0f, 4.0f}, new float[]{3.0f, 4.0f}),
                1.0e-6f);
        assertEquals(0.0f,
                VectorMathUtils.cosineSimilarity(new float[]{1.0f, 0.0f}, new float[]{0.0f, 1.0f}),
                1.0e-6f);
        assertEquals(-1.0f,
                VectorMathUtils.cosineSimilarity(new float[]{1.0f, 0.0f}, new float[]{-1.0f, 0.0f}),
                1.0e-6f);
    }

    @Test
    void outerProductUsesRowMajorInputOrder() {
        float[] actual = VectorMathUtils.outerProduct(
                new float[]{1.0f, -2.0f},
                new float[]{3.0f, 4.0f, -5.0f});

        assertArrayEquals(new float[]{3.0f, 4.0f, -5.0f, -6.0f, -8.0f, 10.0f}, actual);
    }

    @Test
    void outerProductOfEmptyInputIsEmpty() {
        assertArrayEquals(new float[0], VectorMathUtils.outerProduct(new float[0], new float[]{1.0f}));
        assertArrayEquals(new float[0], VectorMathUtils.outerProduct(new float[]{1.0f}, new float[0]));
    }

    @Test
    void outerProductDoesNotMutateItsInputs() {
        float[] xs = {1.0f, -2.0f};
        float[] ys = {3.0f, 4.0f};

        VectorMathUtils.outerProduct(xs, ys);

        assertArrayEquals(new float[]{1.0f, -2.0f}, xs);
        assertArrayEquals(new float[]{3.0f, 4.0f}, ys);
    }

    @Test
    void precomputeFreqsCisReturnsPositionMajorCosineSinePairs() {
        float[][] actual = VectorMathUtils.precomputeFreqsCis(4, 2, 4.0, 2.0);

        assertEquals(4, actual.length);
        assertPairEquals(0.0, actual[0]);
        assertPairEquals(0.0, actual[1]);
        assertPairEquals(0.5, actual[2]);
        assertPairEquals(0.25, actual[3]);
    }

    @Test
    void precomputeFreqsCisReturnsNoRowsForAnEmptyPositionRange() {
        assertEquals(0, VectorMathUtils.precomputeFreqsCis(4, 0, 10_000.0, 1.0).length);
    }

    @Test
    void cosineSimilarityMeasuresDirection() {
        assertEquals(1.0f,
                VectorMathUtils.cosineSimilarity(new float[]{2.0f, 4.0f}, new float[]{1.0f, 2.0f}),
                1.0e-6f);
        assertEquals(0.0f,
                VectorMathUtils.cosineSimilarity(new float[]{1.0f, 0.0f}, new float[]{0.0f, 3.0f}),
                1.0e-6f);
        assertEquals(-1.0f,
                VectorMathUtils.cosineSimilarity(new float[]{1.0f, 2.0f}, new float[]{-2.0f, -4.0f}),
                1.0e-6f);
    }

    @Test
    void l2NormalizesInPlace() {
        float[] values = {3.0f, 4.0f};

        VectorMathUtils.l2normalize(values);

        assertArrayEquals(new float[]{0.6f, 0.8f}, values, 1.0e-6f);
    }

    @Test
    void l1NormalizesUsingAbsoluteMagnitude() {
        float[] values = {-2.0f, 1.0f, 1.0f};

        VectorMathUtils.l1normalize(values);

        assertArrayEquals(new float[]{-0.5f, 0.25f, 0.25f}, values, 1.0e-6f);
    }

    @Test
    void l2normalizeMutatesVectorToUnitLength() {
        float[] vector = {3.0f, -4.0f};

        VectorMathUtils.l2normalize(vector);

        assertArrayEquals(new float[]{0.6f, -0.8f}, vector, 1.0e-6f);
        assertEquals(1.0f, VectorMathUtils.cosineSimilarity(vector, vector), 1.0e-6f);
    }

    @Test
    void l1normalizeMutatesVectorToUnitAbsoluteSum() {
        float[] vector = {-2.0f, 1.0f, 3.0f};

        VectorMathUtils.l1normalize(vector);

        assertArrayEquals(new float[]{-1.0f / 3.0f, 1.0f / 6.0f, 0.5f}, vector, 1.0e-6f);
        assertEquals(1.0f, Math.abs(vector[0]) + Math.abs(vector[1]) + Math.abs(vector[2]), 1.0e-6f);
    }

    @Test
    void normalizingAnEmptyVectorLeavesItEmpty() {
        float[] l1 = {};
        float[] l2 = {};

        VectorMathUtils.l1normalize(l1);
        VectorMathUtils.l2normalize(l2);

        assertArrayEquals(new float[0], l1);
        assertArrayEquals(new float[0], l2);
    }

    private static void assertPairEquals(double angle, float[] actual) {
        assertEquals((float) Math.cos(angle), actual[0], 1.0e-6f);
        assertEquals((float) Math.sin(angle), actual[1], 1.0e-6f);
    }
}
