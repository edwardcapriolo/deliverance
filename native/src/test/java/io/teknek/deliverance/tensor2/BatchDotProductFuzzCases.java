package io.teknek.deliverance.tensor2;

import org.junit.jupiter.params.provider.Arguments;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

final class BatchDotProductFuzzCases {
    private static final long SEED = 0xbadc0ffeeL;

    private BatchDotProductFuzzCases() {
    }

    static Stream<Arguments> cases() {
        List<Case> cases = new ArrayList<>();
        int id = 0;
        int[] resultRows = {1, 2, 3, 5, 8, 13};
        int[] bRows = {1, 2, 3, 5, 8, 16, 31, 64, 128};
        int[] columns = {1, 2, 3, 7, 16, 31, 32, 33, 64, 95, 96, 127, 128, 129, 256};
        int[] aRowOffsets = {0, 1, 3, 7};
        int[] columnOffsets = {0, 1, 5, 16, 31, 32};
        for (int rows : resultRows) {
            for (int bRowCount : bRows) {
                for (int k : columns) {
                    int aRowOffset = aRowOffsets[id % aRowOffsets.length];
                    int aColumnOffset = columnOffsets[id % columnOffsets.length];
                    int bColumnOffset = columnOffsets[(id + 2) % columnOffsets.length];
                    int bRowOffset = id % 3;
                    int resultRowOffset = id % 2;
                    cases.add(new Case("fixed_rows_" + rows + "_brows_" + bRowCount + "_k_" + k,
                            rows, aRowOffset, aColumnOffset, bColumnOffset, k, resultRowOffset,
                            bRowOffset, bRowCount, id++));
                }
            }
        }

        Random random = new Random(SEED);
        for (int i = 0; i < 96; i++) {
            int rows = resultRows[random.nextInt(resultRows.length)];
            int k = columns[random.nextInt(columns.length)];
            int aColumnOffset = columnOffsets[random.nextInt(columnOffsets.length)];
            int bColumnOffset = columnOffsets[random.nextInt(columnOffsets.length)];
            int rowChunkSize = bRows[random.nextInt(bRows.length)];
            cases.add(new Case("fuzz_" + i, rows, random.nextInt(8), aColumnOffset, bColumnOffset, k,
                    random.nextInt(4), random.nextInt(5), rowChunkSize, random.nextInt()));
        }
        return cases.stream().map(Arguments::of);
    }

    record Case(String name, int resultRows, int aRowOffset, int aColumnOffset, int bColumnOffset,
            int columnLength, int resultRowOffset, int bRowOffset, int rowChunkSize, int seed) {
        int aRows() {
            return aRowOffset + resultRows;
        }

        int aColumns() {
            return aColumnOffset + columnLength;
        }

        int bRows() {
            return bRowOffset + rowChunkSize;
        }

        int bColumns() {
            return bColumnOffset + columnLength;
        }

        int resultColumns() {
            return resultRowOffset + bRowOffset + rowChunkSize;
        }

        @Override
        public String toString() {
            return name + "[resultRows=" + resultRows + ", aRowOffset=" + aRowOffset
                    + ", aColumnOffset=" + aColumnOffset + ", bColumnOffset=" + bColumnOffset
                    + ", columnLength=" + columnLength + ", resultRowOffset=" + resultRowOffset
                    + ", bRowOffset=" + bRowOffset + ", rowChunkSize=" + rowChunkSize + "]";
        }
    }
}
