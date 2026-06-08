package io.teknek.deliverance.tensor.operations;

import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorDisplayUtil;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class TensorOperationsSaxpyTest {

    @Test
    public void saxpyUsesAlphaAndInputRowOffsetsForBatchWindow() {
        TensorOperations ops = new NaiveTensorOperations();
        try (AbstractTensor alpha = new FloatBufferTensor(1, 5);
             AbstractTensor x = new FloatBufferTensor(4, 3);
             AbstractTensor y = new FloatBufferTensor(1, 3)) {
            for (int i = 0; i < 5; i++) {
                alpha.set(i + 1, 0, i);
            }
            int value = 1;
            for (int row = 0; row < 4; row++) {
                for (int col = 0; col < 3; col++) {
                    x.set(value++, row, col);
                }
            }

            ops.saxpy(alpha, x, y, 0, 0, 3, 2, 1, 2);

            assertEquals("[0][0]= 40.0000 [0][1]= 47.0000 [0][2]= 54.0000".trim(),
                    TensorDisplayUtil.pretty2dDisplayAll(y).trim());
        }
    }
}
