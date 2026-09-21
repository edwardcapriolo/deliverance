package io.teknek.deliverance.tensor.operations;

import org.junit.jupiter.params.provider.Arguments;

import java.util.stream.Stream;

public final class SaxpyCases {
    private SaxpyCases() {
    }

    public static Stream<Arguments> scalarCases() {
        return Stream.of(
                Arguments.of(new ScalarCase(0, 0, 32)),
                Arguments.of(new ScalarCase(3, 5, 31)),
                Arguments.of(new ScalarCase(2, 1, 9))
        );
    }

    public static Stream<Arguments> batchCases() {
        return Stream.of(
                Arguments.of(new BatchCase(0, 0, 32, 0, 0, 4)),
                Arguments.of(new BatchCase(2, 3, 31, 1, 2, 5)),
                Arguments.of(new BatchCase(1, 4, 9, 3, 1, 7)),
                Arguments.of(new BatchCase(0, 2, 128, 0, 0, 32))
        );
    }

    public record ScalarCase(int xOffset, int yOffset, int length) {
        @Override
        public String toString() {
            return "xoffset=" + xOffset + " yoffset=" + yOffset + " limit=" + length;
        }
    }

    public record BatchCase(int xOffset, int yOffset, int length, int alphaOffset, int xRowOffset,
            int batchSize) {
        @Override
        public String toString() {
            return "xoffset=" + xOffset + " yoffset=" + yOffset + " limit=" + length
                    + " aOffset=" + alphaOffset + " xRowOffset=" + xRowOffset + " batch=" + batchSize;
        }
    }
}
