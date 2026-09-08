package io.teknek.deliverance.tensor2;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

class TensorRefBorrowedTest {

    @Test
    void borrowedTensorRefDoesNotReturnToAllocatorOnClose() throws Exception {
        FloatBufferTensor tensor = new FloatBufferTensor(1, 2);
        TensorRef ref = TensorRef.borrowed(tensor);

        ref.close();

        tensor.set(3.0f, 0, 1);
        assertEquals(3.0f, tensor.get(0, 1));
    }

    @Test
    void lighterCanUseBorrowedF32Tensor() {
        FloatBufferTensor a = new FloatBufferTensor(1, 3);
        FloatBufferTensor b = new FloatBufferTensor(1, 3);
        a.set(2.0f, 0, 0);
        a.set(3.0f, 0, 1);
        a.set(4.0f, 0, 2);
        b.set(10.0f, 0, 0);
        b.set(20.0f, 0, 1);
        b.set(30.0f, 0, 2);
        Lighter lighter = new Lighter(new MetricRegistry(), Map.of(
                TensorProviderKind.PANAMA, new PanamaOps(),
                TensorProviderKind.NAIVE, new NaiveOps()
        ));

        lighter.multiplyAccumulate(new MultiplyAccumulate(TensorRef.borrowed(b))
                .into(TensorRef.borrowed(a))
                .offsetAndLength(0, 3));

        assertEquals(20.0f, a.get(0, 0));
        assertEquals(60.0f, a.get(0, 1));
        assertEquals(120.0f, a.get(0, 2));
    }
}
