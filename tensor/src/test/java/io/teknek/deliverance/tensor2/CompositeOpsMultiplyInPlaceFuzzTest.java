package io.teknek.deliverance.tensor2;

import io.dropwizard.metrics5.MetricName;
import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CompositeOpsMultiplyInPlaceFuzzTest {
    private record Case(String name, DType dType, int rows, int columns, int offset, int length, float factor,
            long seed) {
        @Override
        public String toString() {
            return name;
        }
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("cases")
    void multiplyInPlaceMatchesReference(Case c) {
        MetricRegistry metrics = new MetricRegistry();
        Lighter lighter = new Lighter(metrics);
        CompositeOps ops = new CompositeOps(lighter, metrics);
        try (TensorRef target = lighter.allocate(c.dType(), TensorShape.of(c.rows(), c.columns()))) {
            float[][] expected = fill(target, c.seed());

            for (int row = 0; row < c.rows(); row++) {
                for (int column = c.offset(); column < c.offset() + c.length(); column++) {
                    expected[row][column] *= c.factor();
                }
            }

            ops.multiplyInPlace(new MultiplyInPlace(c.factor()).target(target).offsetAndLength(c.offset(), c.length()),
                    java.util.Map.of("case", c.name()));

            for (int row = 0; row < c.rows(); row++) {
                for (int column = 0; column < c.columns(); column++) {
                    float tolerance = c.dType() == DType.BF16 ? bf16Tolerance(expected[row][column]) : 1.0e-5f;
                    assertEquals(expected[row][column], target.underlying().get(row, column), tolerance,
                            c + " row=" + row + " column=" + column);
                }
            }
        }
        assertEquals(1, metrics.meter(new MetricName("tensor2.composite.multiply_in_place",
                java.util.Map.of("case", c.name(), CompositeOps.TENSOR_TYPE, c.dType().name(), CompositeOps.LENGTH,
                        String.valueOf(c.length())))).getCount());
    }

    static Stream<Case> cases() {
        List<Case> cases = new ArrayList<>();
        float[] factors = {0.0f, 1.0f, -1.0f, 0.125f, -0.75f, 3.5f};
        int index = 0;
        for (DType dType : List.of(DType.F32, DType.BF16)) {
            for (int[] shape : List.of(new int[] {1, 1}, new int[] {1, 17}, new int[] {2, 33},
                    new int[] {4, 65})) {
                for (float factor : factors) {
                    int columns = shape[1];
                    int offset = columns == 1 ? 0 : index % Math.max(1, columns / 3);
                    int maxLength = columns - offset;
                    int length = 1 + (index * 7 % maxLength);
                    cases.add(new Case("edge_" + index, dType, shape[0], columns, offset, length, factor,
                            1000L + index));
                    index++;
                }
            }
        }
        Random random = new Random(4511);
        for (int i = 0; i < 80; i++) {
            DType dType = i % 2 == 0 ? DType.F32 : DType.BF16;
            int rows = 1 + random.nextInt(5);
            int columns = 1 + random.nextInt(130);
            int offset = random.nextInt(columns);
            int length = 1 + random.nextInt(columns - offset);
            float factor = (random.nextFloat() - 0.5f) * 8.0f;
            cases.add(new Case("fuzz_" + i, dType, rows, columns, offset, length, factor, random.nextLong()));
        }
        return cases.stream();
    }

    private static float[][] fill(TensorRef target, long seed) {
        Random random = new Random(seed);
        float[][] expected = new float[target.shape().first()][(int) target.shape().last()];
        for (int row = 0; row < target.shape().first(); row++) {
            for (int column = 0; column < target.shape().last(); column++) {
                float value = (random.nextFloat() - 0.5f) * 20.0f;
                target.underlying().set(value, row, column);
                expected[row][column] = target.underlying().get(row, column);
            }
        }
        return expected;
    }

    private static float bf16Tolerance(float expected) {
        return Math.max(0.08f, Math.abs(expected) * 0.01f);
    }
}
