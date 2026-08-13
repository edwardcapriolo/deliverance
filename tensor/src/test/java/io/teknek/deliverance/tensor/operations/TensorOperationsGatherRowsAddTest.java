package io.teknek.deliverance.tensor.operations;

import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class TensorOperationsGatherRowsAddTest {

    @Test
    void gatherRowsAddUsesAllIdArraysAndHonorsRowRange() {
        TensorOperations ops = new NaiveTensorOperations();
        try (AbstractTensor output = new FloatBufferTensor(4, 3);
             AbstractTensor word = matrix(7, 3, 100.0f);
             AbstractTensor tokenType = matrix(5, 3, 10.0f);
             AbstractTensor position = matrix(9, 3, 1.0f)) {
            int[] inputIds = { 2, 3, 4, 5 };
            int[] tokenTypeIds = { 1, 2, 3, 4 };
            int[] positionIds = { 6, 5, 4, 3 };

            ops.gatherRowsAdd(output, word, inputIds, tokenType, tokenTypeIds, position, positionIds, 1, 2);

            for (int col = 0; col < 3; col++) {
                assertEquals(0.0f, output.get(0, col));
                assertEquals(0.0f, output.get(3, col));
                assertEquals(word.get(3, col) + tokenType.get(2, col) + position.get(5, col),
                        output.get(1, col));
                assertEquals(word.get(4, col) + tokenType.get(3, col) + position.get(4, col),
                        output.get(2, col));
            }
        }
    }

    private static AbstractTensor matrix(int rows, int cols, float scale) {
        AbstractTensor tensor = new FloatBufferTensor(rows, cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                tensor.set(scale + row + col / 10.0f, row, col);
            }
        }
        return tensor;
    }
}
