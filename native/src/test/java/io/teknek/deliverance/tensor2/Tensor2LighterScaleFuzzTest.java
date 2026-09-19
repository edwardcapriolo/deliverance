package io.teknek.deliverance.tensor2;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;

class Tensor2LighterScaleFuzzTest {
    private record Case(String name, DType dType, int rows, int columns, int offset, int length, float factor,
            long seed) {
        @Override
        public String toString() {
            return name;
        }
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("cases")
    void nativeScaleMatchesReference(Case c) {
        MetricRegistry metrics = new MetricRegistry();
        Lighter lighter = new Lighter(metrics);
        lighter.putTensorOperations(TensorProviderKind.SIMD, new NativeOps());
        try (TensorRef target = lighter.allocate(c.dType(), TensorShape.of(c.rows(), c.columns()))) {
            float[][] expected = fill(target, c.seed());
            for (int row = 0; row < c.rows(); row++) {
                for (int column = c.offset(); column < c.offset() + c.length(); column++) {
                    expected[row][column] *= c.factor();
                }
            }

            lighter.scale(new Scale(c.factor()).target(target).offsetAndLength(c.offset(), c.length()),
                    Map.of("case", c.name()));

            for (int row = 0; row < c.rows(); row++) {
                for (int column = 0; column < c.columns(); column++) {
                    float tolerance = c.dType() == DType.BF16 ? bf16Tolerance(expected[row][column]) : 1.0e-5f;
                    assertEquals(expected[row][column], target.underlying().get(row, column), tolerance,
                            c + " row=" + row + " column=" + column);
                }
            }
        }
    }

    static Stream<Case> cases() {
        List<Case> cases = new ArrayList<>();
        int index = 0;
        for (DType dType : List.of(DType.F32, DType.BF16)) {
            for (int[] shape : List.of(new int[] {1, 1}, new int[] {1, 16}, new int[] {2, 17},
                    new int[] {3, 65}, new int[] {4, 129})) {
                for (float factor : List.of(0.0f, 1.0f, -1.0f, 0.25f, -2.5f, 3.75f)) {
                    int columns = shape[1];
                    int offset = columns == 1 ? 0 : index % Math.max(1, columns / 2);
                    int length = Math.max(1, columns - offset - (index % Math.min(columns - offset, 5)));
                    cases.add(new Case("edge_" + dType + "_" + index, dType, shape[0], columns, offset, length,
                            factor, 2000L + index));
                    index++;
                }
            }
        }
        Random random = new Random(8712);
        for (int i = 0; i < 80; i++) {
            DType dType = i % 2 == 0 ? DType.F32 : DType.BF16;
            int rows = 1 + random.nextInt(5);
            int columns = 1 + random.nextInt(257);
            int offset = random.nextInt(columns);
            int length = 1 + random.nextInt(columns - offset);
            float factor = (random.nextFloat() - 0.5f) * 10.0f;
            cases.add(new Case("fuzz_" + dType + "_" + i, dType, rows, columns, offset, length, factor,
                    random.nextLong()));
        }
        return cases.stream();
    }

    private static float[][] fill(TensorRef target, long seed) {
        Random random = new Random(seed);
        float[][] expected = new float[target.shape().first()][(int) target.shape().last()];
        for (int row = 0; row < target.shape().first(); row++) {
            for (int column = 0; column < target.shape().last(); column++) {
                float value = (random.nextFloat() - 0.5f) * 40.0f;
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
