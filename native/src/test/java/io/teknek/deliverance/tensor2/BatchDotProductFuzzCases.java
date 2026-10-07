package io.teknek.deliverance.tensor2;

import org.junit.jupiter.params.provider.Arguments;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

public final class BatchDotProductFuzzCases {
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

    public static Stream<Arguments> sharedI8Q4Cases() {
        List<Case> cases = new ArrayList<>();
        int id = 0;
        int[] resultRows = {1, 2, 3, 5, 8, 13};
        int[] rowChunks = {1, 2, 3, 5, 8, 16, 31, 64, 128};
        int[] lengths = {32, 64, 96, 128, 160, 256, 768, 1024};
        int[] offsets = {0, 32, 64};
        for (int resultRowCount : resultRows) {
            for (int rowChunkSize : rowChunks) {
                for (int length : lengths) {
                    int aColumnOffset = offsets[id % offsets.length];
                    int bColumnOffset = offsets[(id + 1) % offsets.length];
                    int bRowOffset = id % 5;
                    cases.add(new Case("shared_i8_q4_" + id, resultRowCount, 0,
                            aColumnOffset, bColumnOffset, length, bRowOffset + id % 3, bRowOffset,
                            rowChunkSize, id++));
                }
            }
        }
        Random random = new Random(0x8a4f19L);
        for (int i = 0; i < 128; i++) {
            int resultRowCount = resultRows[random.nextInt(resultRows.length)];
            int rowChunkSize = rowChunks[random.nextInt(rowChunks.length)];
            int length = lengths[random.nextInt(lengths.length)];
            int bRowOffset = random.nextInt(5);
            cases.add(new Case("shared_i8_q4_random_" + i, resultRowCount, 0,
                    offsets[random.nextInt(offsets.length)], offsets[random.nextInt(offsets.length)], length,
                    bRowOffset + random.nextInt(4), bRowOffset, rowChunkSize, random.nextInt()));
        }
        return cases.stream().map(Arguments::of);
    }

    public static float inputValue(int row, int column, int seed) {
        return ((row * 17 + column * 31 + seed) % 257 - 128) / 64.0f;
    }

    public static float weightValue(int row, int column, int seed) {
        return ((row * 13 + column * 19 + seed) % 251 - 125) / 96.0f;
    }

    static Stream<Arguments> q8Cases() {
        List<Case> cases = new ArrayList<>();
        int id = 0;
        int[] resultRows = {1, 2, 3, 5, 8};
        int[] bRows = {1, 2, 3, 5, 8, 16, 31, 64};
        int[] columns = {32, 64, 96, 128, 256};
        int[] columnOffsets = {0, 32, 64};
        for (int rows : resultRows) {
            for (int bRowCount : bRows) {
                for (int k : columns) {
                    int aColumnOffset = columnOffsets[id % columnOffsets.length];
                    int bColumnOffset = columnOffsets[(id + 1) % columnOffsets.length];
                    int bRowOffset = id % 3;
                    int resultRowOffset = id % 2;
                    cases.add(new Case("q8_fixed_rows_" + rows + "_brows_" + bRowCount + "_k_" + k,
                            rows, id % 5, aColumnOffset, bColumnOffset, k, resultRowOffset,
                            bRowOffset, bRowCount, id++));
                }
            }
        }
        Random random = new Random(SEED ^ 0x51deca5eL);
        for (int i = 0; i < 96; i++) {
            int rows = resultRows[random.nextInt(resultRows.length)];
            int k = columns[random.nextInt(columns.length)];
            int aColumnOffset = columnOffsets[random.nextInt(columnOffsets.length)];
            int bColumnOffset = columnOffsets[random.nextInt(columnOffsets.length)];
            int rowChunkSize = bRows[random.nextInt(bRows.length)];
            cases.add(new Case("q8_fuzz_" + i, rows, random.nextInt(6), aColumnOffset, bColumnOffset, k,
                    random.nextInt(4), random.nextInt(5), rowChunkSize, random.nextInt()));
        }
        return cases.stream().map(Arguments::of);
    }

    public record Case(String name, int resultRows, int aRowOffset, int aColumnOffset, int bColumnOffset,
            int columnLength, int resultRowOffset, int bRowOffset, int rowChunkSize, int seed) {
        public int aRows() {
            return aRowOffset + resultRows;
        }

        public int aColumns() {
            return aColumnOffset + columnLength;
        }

        public int bRows() {
            return bRowOffset + rowChunkSize;
        }

        public int bColumns() {
            return bColumnOffset + columnLength;
        }

        public int resultColumns() {
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
