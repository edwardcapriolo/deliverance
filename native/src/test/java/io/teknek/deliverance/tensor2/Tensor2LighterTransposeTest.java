package io.teknek.deliverance.tensor2;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;

class Tensor2LighterTransposeTest {
    @ParameterizedTest(name = "{0}")
    @MethodSource("cases")
    void transposeMaterializesReversedShape(Case c) {
        Lighter lighter = new Lighter(new MetricRegistry());
        int[] reversed = reverse(c.shape());
        try (TensorRef source = lighter.allocate(c.dtype(), TensorShape.of(c.shape()));
             TensorRef destination = lighter.allocate(c.dtype(), TensorShape.of(reversed))) {
            int[] cursor = new int[c.shape().length];
            do {
                source.set(value(cursor, c.seed()), cursor);
            } while (advance(cursor, c.shape()));

            lighter.transpose(new Transpose(source).destination(destination));

            cursor = new int[c.shape().length];
            do {
                int[] destinationCursor = reverse(cursor);
                assertEquals(value(cursor, c.seed()), destination.get(destinationCursor),
                        c + " source=" + java.util.Arrays.toString(cursor));
            } while (advance(cursor, c.shape()));
        }
    }

    static Stream<Arguments> cases() {
        List<Case> cases = new ArrayList<>();
        cases.add(new Case("matrix", DType.F32, new int[]{3, 2}, 7));
        cases.add(new Case("singleton", DType.F32, new int[]{1, 4, 1}, 11));
        cases.add(new Case("bf16", DType.BF16, new int[]{2, 3, 4}, 19));
        Random random = new Random(0x7a11ceL);
        for (int i = 0; i < 32; i++) {
            int rank = 2 + random.nextInt(3);
            int[] shape = new int[rank];
            for (int dimension = 0; dimension < rank; dimension++) {
                shape[dimension] = 1 + random.nextInt(5);
            }
            cases.add(new Case("random_" + i, i % 2 == 0 ? DType.F32 : DType.BF16, shape, random.nextInt()));
        }
        return cases.stream().map(Arguments::of);
    }

    private static float value(int[] coordinates, int seed) {
        int value = seed;
        for (int coordinate : coordinates) {
            value = value * 31 + coordinate;
        }
        return (value % 257) / 32.0f;
    }

    private static int[] reverse(int[] coordinates) {
        int[] reversed = new int[coordinates.length];
        for (int i = 0; i < coordinates.length; i++) {
            reversed[i] = coordinates[coordinates.length - i - 1];
        }
        return reversed;
    }

    private static boolean advance(int[] cursor, int[] shape) {
        for (int dimension = cursor.length - 1; dimension >= 0; dimension--) {
            if (++cursor[dimension] < shape[dimension]) {
                return true;
            }
            cursor[dimension] = 0;
        }
        return false;
    }

    private record Case(String name, DType dtype, int[] shape, int seed) {
        @Override
        public String toString() {
            return name + java.util.Arrays.toString(shape) + "-" + dtype;
        }
    }
}
