package io.teknek.deliverance.tensor2;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.Map;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** Provider-specific coverage for every CompositeOps operation. */
class CompositeOpsProviderFuzzTest {
    private record Provider(String name, TensorProviderKind kind, TensorOps ops, boolean enabled) {
    }

    @ParameterizedTest(name = "multiply slice {0}")
    @MethodSource("providers")
    void multiplyInPlaceSupportsSlices(Provider provider) {
        assume(provider);
        Lighter lighter = strict(provider);
        try (TensorRef parent = lighter.allocate(DType.F32, TensorShape.of(3, 17));
             TensorRef target = parent.slice(1)) {
            fill(parent, 11);
            float[][] expected = values(parent);
            for (int column = 2; column < 15; column++) {
                expected[1][column] *= 0.25f;
            }
            new CompositeOps(lighter).multiplyInPlace(new MultiplyInPlace(0.25f)
                    .target(target).offsetAndLength(2, 13));
            assertValues(parent, expected, 0.0f);
        }
    }

    @ParameterizedTest(name = "softmax slice {0}")
    @MethodSource("providers")
    void scaledSoftMaxSupportsSlices(Provider provider) {
        assume(provider);
        Lighter lighter = strict(provider);
        try (TensorRef parent = lighter.allocate(DType.F32, TensorShape.of(3, 17));
             TensorRef target = parent.slice(1)) {
            fill(parent, 17);
            float[][] expected = values(parent);
            float max = Float.NEGATIVE_INFINITY;
            for (int column = 2; column < 15; column++) {
                max = Math.max(max, expected[1][column]);
            }
            float sum = 0.0f;
            for (int column = 2; column < 15; column++) {
                sum += expected[1][column] = (float) Math.exp(expected[1][column] - max);
            }
            for (int column = 2; column < 15; column++) {
                expected[1][column] /= sum;
            }
            new CompositeOps(lighter).scaledSoftMax(new ScaledSoftMax(1.0f)
                    .target(target).offsetAndLength(2, 13));
            assertValues(parent, expected, 1.0e-5f);
        }
    }

    @ParameterizedTest(name = "silu {0}")
    @MethodSource("providers")
    void siluMatchesReference(Provider provider) {
        assume(provider);
        Lighter lighter = strict(provider);
        try (TensorRef target = lighter.allocate(DType.F32, TensorShape.of(3, 17))) {
            fill(target, 23);
            float[][] expected = values(target);
            for (float[] row : expected) {
                for (int column = 0; column < row.length; column++) {
                    row[column] = row[column] / (1.0f + (float) Math.exp(-row[column]));
                }
            }
            new CompositeOps(lighter).silu(target);
            assertValues(target, expected, 1.0e-5f);
        }
    }

    @ParameterizedTest(name = "residual {0}")
    @MethodSource("providers")
    void residualMatchesReference(Provider provider) {
        assume(provider);
        Lighter lighter = strict(provider);
        try (TensorRef target = lighter.allocate(DType.F32, TensorShape.of(1, 17));
             TensorRef residual = lighter.allocate(DType.F32, TensorShape.of(1, 17))) {
            fill(target, 29);
            fill(residual, 31);
            float[][] expected = values(target);
            float[][] addition = values(residual);
            for (int column = 0; column < expected[0].length; column++) {
                expected[0][column] = expected[0][column] * 0.75f + addition[0][column];
            }
            new CompositeOps(lighter).addResidual(target, residual, 0.75f);
            assertValues(target, expected, 1.0e-5f);
        }
    }

    @ParameterizedTest(name = "rms norm {0}")
    @MethodSource("providers")
    void rmsNormMatchesReference(Provider provider) {
        assume(provider);
        Lighter lighter = strict(provider);
        try (TensorRef input = lighter.allocate(DType.F32, TensorShape.of(2, 16));
             TensorRef output = lighter.allocate(DType.F32, TensorShape.of(2, 16));
             TensorRef weights = lighter.allocate(DType.F32, TensorShape.of(1, 16))) {
            fill(input, 37);
            fill(weights, 41);
            new CompositeOps(lighter).rmsNorm(output, input, weights, 1.0e-6f, 0.0f);
            for (int row = 0; row < 2; row++) {
                double sum = 0.0;
                for (int column = 0; column < 16; column++) {
                    float value = input.get(row, column);
                    sum += value * value;
                }
                float scale = (float) (1.0 / Math.sqrt(sum / 16.0 + 1.0e-6));
                for (int column = 0; column < 16; column++) {
                    assertEquals(input.get(row, column) * scale * weights.get(0, column),
                            output.get(row, column), 1.0e-5f);
                }
            }
        }
    }

    @ParameterizedTest(name = "grouped rms norm {0}")
    @MethodSource("providers")
    void groupedRmsNormMatchesReference(Provider provider) {
        assume(provider);
        Lighter lighter = strict(provider);
        try (TensorRef target = lighter.allocate(DType.F32, TensorShape.of(2, 16));
             TensorRef weights = lighter.allocate(DType.F32, TensorShape.of(1, 4))) {
            fill(target, 43);
            fill(weights, 47);
            float[][] expected = values(target);
            for (int row = 0; row < 2; row++) {
                for (int group = 0; group < 4; group++) {
                    int offset = group * 4;
                    double sum = 0.0;
                    for (int column = 0; column < 4; column++) {
                        float value = expected[row][offset + column];
                        sum += value * value;
                    }
                    double scale = 1.0 / Math.sqrt(sum / 4.0 + 1.0e-6);
                    for (int column = 0; column < 4; column++) {
                        expected[row][offset + column] = (float) (expected[row][offset + column] * scale
                                * weights.get(0, column));
                    }
                }
            }
            new CompositeOps(lighter).groupedRmsNormInPlace(target, 4, 4, 1.0e-6f, weights);
            assertValues(target, expected, 1.0e-5f);
        }
    }

    @ParameterizedTest(name = "rotary embedding {0}")
    @MethodSource("providers")
    void rotaryEmbeddingMatchesReference(Provider provider) {
        assume(provider);
        Lighter lighter = strict(provider);
        int headSize = 4;
        float[][] frequencies = new float[16][2];
        for (int position = 0; position < frequencies.length; position++) {
            frequencies[position][0] = (float) Math.cos(position * 0.17);
            frequencies[position][1] = (float) Math.sin(position * 0.17);
        }
        try (TensorRef target = lighter.allocate(DType.F32, TensorShape.of(3, headSize))) {
            fill(target, 53);
            float[][] expected = values(target);
            for (int row = 0; row < 3; row++) {
                int frequencyOffset = (2 + row) * (headSize / 2);
                for (int dimension = 0; dimension < headSize / 2; dimension++) {
                    float x0 = expected[row][dimension];
                    float x1 = expected[row][dimension + headSize / 2];
                    float cosine = frequencies[frequencyOffset + dimension][0];
                    float sine = frequencies[frequencyOffset + dimension][1];
                    expected[row][dimension] = x0 * cosine - x1 * sine;
                    expected[row][dimension + headSize / 2] = x0 * sine + x1 * cosine;
                }
            }
            new CompositeOps(lighter).rotaryEmbedding(target, 1, headSize, 2, frequencies);
            assertValues(target, expected, 1.0e-5f);
        }
    }

    static Stream<Arguments> providers() {
        return Stream.of(
                new Provider("naive", TensorProviderKind.NAIVE, new NaiveOps(), true),
                new Provider("panama", TensorProviderKind.PANAMA, new PanamaOps(), true),
                new Provider("simd", TensorProviderKind.SIMD, new NativeOps(), NativeOps.isAvailable()))
                .map(Arguments::of);
    }

    private static Lighter strict(Provider provider) {
        return new Lighter(new MetricRegistry(), Map.of(provider.kind(), provider.ops()));
    }

    private static void assume(Provider provider) {
        Assumptions.assumeTrue(provider.enabled(), provider.name() + " is unavailable");
    }

    private static void fill(TensorRef tensor, int seed) {
        for (int row = 0; row < tensor.shape().first(); row++) {
            for (int column = 0; column < tensor.shape().last(); column++) {
                tensor.set(((row * 13 + column * 7 + seed) % 41 - 20) / 16.0f, row, column);
            }
        }
    }

    private static float[][] values(TensorRef tensor) {
        float[][] values = new float[tensor.shape().first()][tensor.shape().last()];
        for (int row = 0; row < values.length; row++) {
            for (int column = 0; column < values[row].length; column++) {
                values[row][column] = tensor.get(row, column);
            }
        }
        return values;
    }

    private static void assertValues(TensorRef actual, float[][] expected, float tolerance) {
        for (int row = 0; row < expected.length; row++) {
            for (int column = 0; column < expected[row].length; column++) {
                assertEquals(expected[row][column], actual.get(row, column), tolerance,
                        "row=" + row + " column=" + column);
            }
        }
    }
}
