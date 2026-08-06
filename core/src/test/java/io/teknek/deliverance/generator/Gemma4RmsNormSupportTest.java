package io.teknek.deliverance.generator;

import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.impl.BFloat16BufferTensor;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import net.jafama.FastMath;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import static org.junit.jupiter.api.Assertions.assertEquals;

class Gemma4RmsNormSupportTest {

    @Test
    void qwen3ShapedQueryNormMatchesReference() {
        assertQwen3ShapedNormMatchesReference(1, 32, 128, 1.0e-6f, true);
    }

    @Test
    void qwen3ShapedKeyNormMatchesReference() {
        assertQwen3ShapedNormMatchesReference(1, 8, 128, 1.0e-6f, true);
    }

    @Test
    void qwen3ShapedNormWithoutWeightsMatchesReference() {
        assertQwen3ShapedNormMatchesReference(2, 8, 128, 1.0e-6f, false);
    }

    @ParameterizedTest(name = "batch={0} groups={1} groupSize={2} weights={4}")
    @CsvSource({
            "1,32,128,1.0e-6,true",
            "1,8,128,1.0e-6,true",
            "2,8,128,1.0e-6,false"
    })
    void qwen3ShapedSimdNormMatchesScalar(int batchSize, int groups, int groupSize, float eps,
            boolean withWeights) {
        assertSimdMatchesScalar(batchSize, groups, groupSize, eps, withWeights);
    }

    @ParameterizedTest(name = "batch={0} groups={1} groupSize={2} bf16Weights")
    @CsvSource({
            "1,32,128,1.0e-6",
            "1,8,128,1.0e-6",
            "128,32,128,1.0e-6"
    })
    void qwen3ShapedSimdNormMatchesScalarWithBf16Weights(int batchSize, int groups, int groupSize, float eps) {
        try (AbstractTensor scalar = tensor(batchSize, groups * groupSize, 17);
             AbstractTensor simd = tensor(batchSize, groups * groupSize, 17);
             AbstractTensor weights = new BFloat16BufferTensor(weights(groupSize, 29))) {
            Gemma4RmsNormSupport.applyInPlace(scalar, groups, groupSize, eps, weights);
            Gemma4RmsNormSupport.applyInPlaceSimd(simd, groups, groupSize, eps, weights);
            for (int row = 0; row < batchSize; row++) {
                for (int col = 0; col < groups * groupSize; col++) {
                    assertEquals(scalar.get(row, col), simd.get(row, col), 1.0e-6f,
                            "row=" + row + " col=" + col);
                }
            }
        }
    }

    @Test
    void simdNormSupportsUnalignedBf16WeightSliceLikeLoadedModelWeights() {
        int batchSize = 1;
        int groups = 8;
        int groupSize = 128;
        try (AbstractTensor scalar = tensor(batchSize, groups * groupSize, 17);
             AbstractTensor simd = tensor(batchSize, groups * groupSize, 17);
             AbstractTensor backingWeights = new BFloat16BufferTensor(tensor(2, groupSize, 29));
             AbstractTensor unalignedWeights = backingWeights.slice(1)) {
            Gemma4RmsNormSupport.applyInPlace(scalar, groups, groupSize, 1.0e-6f, unalignedWeights);
            Gemma4RmsNormSupport.applyInPlaceSimd(simd, groups, groupSize, 1.0e-6f, unalignedWeights);
            for (int row = 0; row < batchSize; row++) {
                for (int col = 0; col < groups * groupSize; col++) {
                    assertEquals(scalar.get(row, col), simd.get(row, col), 1.0e-6f,
                            "row=" + row + " col=" + col);
                }
            }
        }
    }

    @Test
    void benchmarkQwen3RmsNormShapes() {
        System.out.println("case,groups,groupSize,iterations,scalar_ms,scalar_us,simd_ms,simd_us,speedup");
        bench("query", 32, 128, 50_000);
        bench("key", 8, 128, 50_000);
    }

    private static void bench(String name, int groups, int groupSize, int iterations) {
        try (AbstractTensor scalar = tensor(1, groups * groupSize, 17);
             AbstractTensor simd = tensor(1, groups * groupSize, 17);
             AbstractTensor weights = weights(groupSize, 29)) {
            for (int i = 0; i < 1_000; i++) {
                fill(scalar, 17 + i);
                fill(simd, 17 + i);
                Gemma4RmsNormSupport.applyInPlace(scalar, groups, groupSize, 1.0e-6f, weights);
                Gemma4RmsNormSupport.applyInPlaceSimd(simd, groups, groupSize, 1.0e-6f, weights);
            }
            long start = System.nanoTime();
            for (int i = 0; i < iterations; i++) {
                fill(scalar, 17 + i);
                Gemma4RmsNormSupport.applyInPlace(scalar, groups, groupSize, 1.0e-6f, weights);
            }
            long scalarElapsed = System.nanoTime() - start;

            start = System.nanoTime();
            for (int i = 0; i < iterations; i++) {
                fill(simd, 17 + i);
                Gemma4RmsNormSupport.applyInPlaceSimd(simd, groups, groupSize, 1.0e-6f, weights);
            }
            long simdElapsed = System.nanoTime() - start;

            double scalarMs = scalarElapsed / 1_000_000.0;
            double scalarUs = scalarElapsed / 1_000.0 / iterations;
            double simdMs = simdElapsed / 1_000_000.0;
            double simdUs = simdElapsed / 1_000.0 / iterations;
            System.out.printf(java.util.Locale.ROOT, "%s,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.4f%n", name, groups,
                    groupSize, iterations, scalarMs, scalarUs, simdMs, simdUs, scalarUs / simdUs);
        }
    }

    private static void assertSimdMatchesScalar(int batchSize, int groups, int groupSize, float eps,
            boolean withWeights) {
        try (AbstractTensor scalar = tensor(batchSize, groups * groupSize, 17);
             AbstractTensor simd = tensor(batchSize, groups * groupSize, 17);
             AbstractTensor weights = withWeights ? weights(groupSize, 29) : null) {
            Gemma4RmsNormSupport.applyInPlace(scalar, groups, groupSize, eps, weights);
            Gemma4RmsNormSupport.applyInPlaceSimd(simd, groups, groupSize, eps, weights);
            for (int row = 0; row < batchSize; row++) {
                for (int col = 0; col < groups * groupSize; col++) {
                    assertEquals(scalar.get(row, col), simd.get(row, col), 1.0e-6f,
                            "row=" + row + " col=" + col);
                }
            }
        }
    }

    private static void assertQwen3ShapedNormMatchesReference(int batchSize, int groups, int groupSize, float eps,
            boolean withWeights) {
        try (AbstractTensor actual = tensor(batchSize, groups * groupSize, 17);
             AbstractTensor expected = tensor(batchSize, groups * groupSize, 17);
             AbstractTensor weights = withWeights ? weights(groupSize, 29) : null) {
            applyReference(expected, groups, groupSize, eps, weights);
            Gemma4RmsNormSupport.applyInPlace(actual, groups, groupSize, eps, weights);

            for (int row = 0; row < batchSize; row++) {
                for (int col = 0; col < groups * groupSize; col++) {
                    assertEquals(expected.get(row, col), actual.get(row, col), 1.0e-6f,
                            "row=" + row + " col=" + col);
                }
            }
        }
    }

    private static void applyReference(AbstractTensor tensor, int groups, int groupSize, float eps,
            AbstractTensor weights) {
        int batchSize = (int) tensor.shape().first();
        for (int row = 0; row < batchSize; row++) {
            for (int group = 0; group < groups; group++) {
                int offset = group * groupSize;
                double sumSquares = 0.0;
                for (int col = 0; col < groupSize; col++) {
                    float value = tensor.get(row, offset + col);
                    sumSquares += value * value;
                }
                double invRms = 1.0 / FastMath.sqrt(sumSquares / groupSize + eps);
                for (int col = 0; col < groupSize; col++) {
                    float value = (float) (tensor.get(row, offset + col) * invRms);
                    if (weights != null) {
                        value *= weights.get(0, col);
                    }
                    tensor.set(value, row, offset + col);
                }
            }
        }
    }

    private static AbstractTensor tensor(int rows, int cols, int seed) {
        FloatBufferTensor tensor = new FloatBufferTensor(rows, cols);
        fill(tensor, seed);
        return tensor;
    }

    private static void fill(AbstractTensor tensor, int seed) {
        for (int row = 0; row < tensor.shape().first(); row++) {
            for (int col = 0; col < tensor.shape().last(); col++) {
                tensor.set((((row * 31 + col * 17 + seed) % 47) - 23) / 23.0f, row, col);
            }
        }
    }

    private static AbstractTensor weights(int cols, int seed) {
        FloatBufferTensor tensor = new FloatBufferTensor(1, cols);
        for (int col = 0; col < cols; col++) {
            tensor.set(0.75f + ((((col * 13 + seed) % 19) / 18.0f) * 0.5f), 0, col);
        }
        return tensor;
    }
}
