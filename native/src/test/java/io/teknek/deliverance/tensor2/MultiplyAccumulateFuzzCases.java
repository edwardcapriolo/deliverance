package io.teknek.deliverance.tensor2;

import io.teknek.deliverance.DType;
import org.junit.jupiter.params.provider.Arguments;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

final class MultiplyAccumulateFuzzCases {
    private static final long SEED = 0x51a7e2L;

    private MultiplyAccumulateFuzzCases() {
    }

    static Stream<Arguments> cases() {
        List<Case> cases = new ArrayList<>();
        int id = 0;
        int[] rows = {1, 2, 3, 5};
        int[] columns = {1, 2, 3, 7, 16, 31, 32, 33, 64, 127, 128, 257};
        DType[] dTypes = {DType.F32, DType.BF16};
        for (DType dType : dTypes) {
            for (int aRows : rows) {
                for (int columnCount : columns) {
                    int offset = columnCount == 1 ? 0 : Math.min(columnCount - 1, id % Math.max(1, columnCount / 2));
                    int length = Math.max(1, columnCount - offset);
                    cases.add(new Case("fixed_batch_" + dType + "_" + aRows + "_cols_" + columnCount,
                            dType, aRows, aRows, columnCount, offset, length, id++));
                    if (aRows > 1) {
                        cases.add(new Case("fixed_broadcast_" + dType + "_" + aRows + "_cols_" + columnCount,
                                dType, aRows, 1, columnCount, offset, length, id++));
                    }
                }
            }
        }

        Random random = new Random(SEED);
        for (int i = 0; i < 48; i++) {
            int aRows = rows[random.nextInt(rows.length)];
            int columnCount = columns[random.nextInt(columns.length)];
            int bRows = random.nextBoolean() || aRows == 1 ? aRows : 1;
            int offset = columnCount == 1 ? 0 : random.nextInt(columnCount);
            int length = 1 + random.nextInt(columnCount - offset);
            DType dType = dTypes[random.nextInt(dTypes.length)];
            cases.add(new Case("fuzz_" + dType + "_" + i, dType, aRows, bRows, columnCount, offset, length, random.nextInt()));
        }
        return cases.stream().map(Arguments::of);
    }

    record Case(String name, DType dType, int aRows, int bRows, int columns, int offset, int length, int seed) {
        @Override
        public String toString() {
            return name + "[dType=" + dType + ", aRows=" + aRows + ", bRows=" + bRows + ", columns=" + columns
                    + ", offset=" + offset + ", length=" + length + "]";
        }
    }
}
