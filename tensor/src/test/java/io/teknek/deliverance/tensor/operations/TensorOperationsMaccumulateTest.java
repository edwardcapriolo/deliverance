package io.teknek.deliverance.tensor.operations;

import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.TensorDisplayUtil;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.concurrent.ForkJoinPool;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class TensorOperationsMaccumulateTest {

    @Test
    public void panamaMaccumulateSupportsNonVectorMultipleLength() {
        try (AbstractTensor a = matrix(2, 7, 1.0f);
             AbstractTensor b = matrix(2, 7, 2.0f);
             AbstractTensor expected = new FloatBufferTensor(a);
             AbstractTensor actual = new FloatBufferTensor(a);
             WrappedForkJoinPool pool = new WrappedForkJoinPool(new ForkJoinPool(1))) {
            new NaiveTensorOperations().maccumulate(expected, b, 1, 5);
            new PanamaTensorOperations(MachineSpec.VECTOR_TYPE, Mockito.mock(TensorAllocator.class), pool)
                    .maccumulate(actual, b, 1, 5);

            assertEquals(normalize(TensorDisplayUtil.pretty2dDisplayAll(expected)),
                    normalize(TensorDisplayUtil.pretty2dDisplayAll(actual)));
        }
    }

    @Test
    public void panamaMaccumulateSupportsBroadcastSecondOperand() {
        try (AbstractTensor a = matrix(2, 7, 1.0f);
             AbstractTensor b = matrix(1, 7, 3.0f);
             AbstractTensor expected = new FloatBufferTensor(a);
             AbstractTensor actual = new FloatBufferTensor(a);
             WrappedForkJoinPool pool = new WrappedForkJoinPool(new ForkJoinPool(1))) {
            new NaiveTensorOperations().maccumulate(expected, b, 0, 7);
            new PanamaTensorOperations(MachineSpec.VECTOR_TYPE, Mockito.mock(TensorAllocator.class), pool)
                    .maccumulate(actual, b, 0, 7);

            assertEquals(normalize(TensorDisplayUtil.pretty2dDisplayAll(expected)),
                    normalize(TensorDisplayUtil.pretty2dDisplayAll(actual)));
        }
    }

    private static AbstractTensor matrix(int rows, int cols, float scale) {
        AbstractTensor tensor = new FloatBufferTensor(rows, cols);
        int value = 1;
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                tensor.set(value++ * scale, row, col);
            }
        }
        return tensor;
    }

    private static String normalize(String display) {
        return display.strip().replaceAll("(?m) +$", "");
    }
}
