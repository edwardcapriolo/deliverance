package io.teknek.deliverance.tensor;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.PanamaTensorOperations;
import io.teknek.deliverance.tensor.operations.TensorOperations;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class AbstractTensorUtilsTest {

    @Test
    void quantize() {
        int rows = 4;
        int columns = 32;
        AbstractTensor original = new FloatBufferTensor(rows, columns);
        for (int i = 0; i < rows * columns; i++) {
            original.set(5, 0, i);
        }

        AbstractTensor q4 = AbstractTensorUtils.quantize(original, DType.Q4);
        AbstractTensor i8 = AbstractTensorUtils.quantize(original, DType.I8);
        AbstractTensor bf16 = AbstractTensorUtils.quantize(original, DType.BF16);

        Assertions.assertEquals(5.0, bf16.get(0,4));
        Assertions.assertEquals(5.0, q4.get(0,4));
        Assertions.assertEquals(5.0, i8.get(0,4));
    }
}
