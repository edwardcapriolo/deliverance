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
}
