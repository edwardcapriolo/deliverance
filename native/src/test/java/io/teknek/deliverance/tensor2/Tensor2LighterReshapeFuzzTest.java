package io.teknek.deliverance.tensor2;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.impl.Q4ByteBufferTensor;
import io.teknek.deliverance.tensor.impl.Q8ByteBufferTensor;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;

class Tensor2LighterReshapeFuzzTest {
    @org.junit.jupiter.api.Test
    void f32ToI8ReshapeIntoNonzeroRowSlicePreservesRow() {
        Lighter lighter = panamaOnly();
        try (TensorRef source = lighter.allocate(DType.F32, TensorShape.of(3, 32));
             TensorRef destination = lighter.allocate(DType.I8, TensorShape.of(3, 32));
             TensorRef sourceRow = source.slice(2);
             TensorRef destinationRow = destination.slice(2);
             TensorRef expected = lighter.allocate(DType.F32, TensorShape.of(1, 32))) {
            for (int column = 0; column < 32; column++) {
                for (int row = 0; row < 3; row++) {
                    float value = ((row * 13 + column * 7 + 71) % 41 - 20) / 16.0f;
                    source.set(value, row, column);
                    if (row == 2) {
                        expected.set(value, 0, column);
                    }
                }
            }
            lighter.reshape(sourceRow, destinationRow);
            try (TensorRef expectedI8 = lighter.reshape(expected, DType.I8)) {
                for (int block = 0; block < 32 / Q8Layout.BLOCK_SIZE; block++) {
                    assertEquals(expectedI8.sidecar("q8.scale").get(0, block),
                            destination.sidecar("q8.scale").get(2, block), 0.0f, "block=" + block);
                }
            for (int column = 0; column < 32; column++) {
                    assertEquals(expectedI8.get(0, column), destination.get(2, column), 0.0f,
                        "column=" + column);
                }
            }
        }
    }

    @org.junit.jupiter.api.Test
    void f32ToI8QuantizationIsRowIndependent() {
        Lighter lighter = panamaOnly();
        int columns = 32;
        try (TensorRef batch = lighter.allocate(DType.F32, TensorShape.of(5, columns));
             TensorRef single = lighter.allocate(DType.F32, TensorShape.of(1, columns))) {
            for (int column = 0; column < columns; column++) {
                float value = ((4 * 13 + column * 7 + 91) % 41 - 20) / 16.0f;
                batch.set(value, 4, column);
                single.set(value, 0, column);
            }
            try (TensorRef batchI8 = lighter.reshape(batch, DType.I8);
                 TensorRef singleI8 = lighter.reshape(single, DType.I8)) {
                TensorRef batchScale = batchI8.sidecar("q8.scale");
                TensorRef singleScale = singleI8.sidecar("q8.scale");
                for (int column = 0; column < columns; column++) {
                    assertEquals(singleI8.get(0, column), batchI8.get(4, column), 0.0f,
                            "dequantized row column=" + column);
                }
                assertEquals(singleScale.get(0, 0), batchScale.get(4, 0), 0.0f, "row scale");
            }
        }
    }

    @ParameterizedTest(name = "i8 {0}")
    @MethodSource("cases")
    void f32ToI8MatchesLegacyQ8(Case c) {
        Lighter lighter = panamaOnly();
        try (FloatBufferTensor dense = dense(c);
             TensorRef input = lighter.allocate(DType.F32, TensorShape.of(c.rows(), c.columns()));
             Q8ByteBufferTensor legacy = new Q8ByteBufferTensor(dense)) {
            copy(dense, input);
            try (TensorRef actual = lighter.reshape(input, DType.I8)) {
                I8Tensor tensor = (I8Tensor) actual.underlying();
                TensorRef scale = Q8Layout.scale(actual);
                for (int row = 0; row < c.rows(); row++) {
                    for (int column = 0; column < c.columns(); column++) {
                        int offset = row * c.columns() + column;
                        assertEquals(legacy.getMemorySegment().get(ValueLayout.JAVA_BYTE, offset),
                                tensor.getRawByte(offset), c + " payload offset=" + offset);
                        assertEquals(legacy.get(row, column), actual.get(row, column), 1.0e-6f,
                                c + " row=" + row + " column=" + column);
                    }
                    for (int block = 0; block < c.columns() / Q8Layout.BLOCK_SIZE; block++) {
                        assertEquals(legacy.getBlockF().get(row, block), scale.get(row, block), 0.0f,
                                c + " scale row=" + row + " block=" + block);
                    }
                }
            }
        }
    }

    @ParameterizedTest(name = "q4 {0}")
    @MethodSource("cases")
    void f32ToQ4MatchesLegacyQ4(Case c) {
        Lighter lighter = panamaOnly();
        try (FloatBufferTensor dense = dense(c);
             TensorRef input = lighter.allocate(DType.F32, TensorShape.of(c.rows(), c.columns()));
             Q4ByteBufferTensor legacy = new Q4ByteBufferTensor(dense)) {
            copy(dense, input);
            try (TensorRef actual = lighter.reshape(input, DType.Q4)) {
                Q4Tensor tensor = (Q4Tensor) actual.underlying();
                TensorRef scale = Q4Layout.scale(actual);
                int payloadBytes = c.rows() * c.columns() / 2;
                for (int offset = 0; offset < payloadBytes; offset++) {
                    assertEquals(legacy.getMemorySegment().get(ValueLayout.JAVA_BYTE, offset),
                            tensor.getRawPackedByte(offset), c + " payload offset=" + offset);
                }
                for (int row = 0; row < c.rows(); row++) {
                    for (int column = 0; column < c.columns(); column++) {
                        assertEquals(legacy.get(row, column), actual.get(row, column), 1.0e-6f,
                                c + " row=" + row + " column=" + column);
                    }
                    for (int block = 0; block < c.columns() / Q4Layout.BLOCK_SIZE; block++) {
                        assertEquals(legacy.getBlockF().get(row, block), scale.get(row, block), 0.0f,
                                c + " scale row=" + row + " block=" + block);
                    }
                }
            }
        }
    }

    static Stream<Arguments> cases() {
        List<Case> cases = new ArrayList<>();
        int id = 0;
        int[] rows = {1, 2, 4, 7};
        int[] columns = {32, 64, 96, 128};
        for (int row : rows) {
            for (int column : columns) {
                cases.add(new Case("pattern_" + id, row, column, id++, Pattern.DETERMINISTIC));
                cases.add(new Case("zero_" + id, row, column, id++, Pattern.ZERO));
                cases.add(new Case("negative_max_" + id, row, column, id++, Pattern.NEGATIVE_MAX));
                cases.add(new Case("positive_max_" + id, row, column, id++, Pattern.POSITIVE_MAX));
                cases.add(new Case("alternating_" + id, row, column, id++, Pattern.ALTERNATING));
            }
        }
        Random random = new Random(0x51deca5eL);
        for (int i = 0; i < 64; i++) {
            cases.add(new Case("random_" + i, rows[random.nextInt(rows.length)], columns[random.nextInt(columns.length)],
                    random.nextInt(), Pattern.RANDOM));
        }
        return cases.stream().map(Arguments::of);
    }

    private static Lighter panamaOnly() {
        return new Lighter(new io.dropwizard.metrics5.MetricRegistry(), Map.of(TensorProviderKind.PANAMA, new PanamaOps()));
    }

    private static FloatBufferTensor dense(Case c) {
        FloatBufferTensor tensor = new FloatBufferTensor(TensorShape.of(c.rows(), c.columns()));
        Random random = new Random(c.seed());
        for (int row = 0; row < c.rows(); row++) {
            for (int column = 0; column < c.columns(); column++) {
                tensor.set(value(c, random, row, column), row, column);
            }
        }
        return tensor;
    }

    private static float value(Case c, Random random, int row, int column) {
        return switch (c.pattern()) {
            case ZERO -> 0.0f;
            case NEGATIVE_MAX -> column % Q8Layout.BLOCK_SIZE == 0 ? -2.0f : ((row + column) % 17 - 8) / 16.0f;
            case POSITIVE_MAX -> column % Q8Layout.BLOCK_SIZE == 0 ? 2.0f : ((row + column) % 17 - 8) / 16.0f;
            case ALTERNATING -> (column & 1) == 0 ? 1.0f : -1.0f;
            case RANDOM -> (random.nextFloat() * 2.0f) - 1.0f;
            case DETERMINISTIC -> ((row * 13 + column * 7 + c.seed()) % 41 - 20) / 16.0f;
        };
    }

    private static void copy(FloatBufferTensor source, TensorRef target) {
        for (int row = 0; row < source.shape().first(); row++) {
            for (int column = 0; column < source.shape().last(); column++) {
                target.set(source.get(row, column), row, column);
            }
        }
    }

    private enum Pattern {
        DETERMINISTIC,
        ZERO,
        NEGATIVE_MAX,
        POSITIVE_MAX,
        ALTERNATING,
        RANDOM
    }

    private record Case(String name, int rows, int columns, int seed, Pattern pattern) {
        @Override
        public String toString() {
            return name + "[rows=" + rows + ", columns=" + columns + ", pattern=" + pattern + "]";
        }
    }
}
