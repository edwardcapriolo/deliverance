package io.teknek.deliverance.tensor;

import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class VectorTensorMathUtilsTest {

    /*
    From google ai (trust the answers with your life :)
    import numpy as np
    from scipy.special import logsumexp

    logits = np.array([2.0, 1.0, 0.1])
    # logsumexp provides numerical stability to avoid overflow
    logprobs = logits - logsumexp(logits)

    print(logprobs)
    # Result: array([-0.417, -1.417, -2.317])
    */

    @Test
    public void vectorToVector(){
        int rows = 1;
        int cols = 3;
        AbstractTensor original = new FloatBufferTensor(rows, cols);
        original.set(2.0f, 0, 0);
        original.set(1.0f, 0, 1);
        original.set(0.1f, 0, 2);
        AbstractTensor result = new FloatBufferTensor(rows, cols);
        VectorTensorMathUtils.logSumExpTensor(result, original);
        String origin   = "[0][0]=  2.0000 [0][1]=  1.0000 [0][2]=  0.1000";
        String expected = "[0][0]= -0.4170 [0][1]= -1.4170 [0][2]= -2.3170";
        assertEquals(expected, TensorDisplayUtil.pretty2dDisplayAll(result).trim());
    }

    //https://nhigham.com/2021/01/05/what-is-the-log-sum-exp-function/
    @Test
    public void logSumExpTest(){
        int rows =  1;
        int cols = 3;
        AbstractTensor original = new FloatBufferTensor(rows, cols);
        original.set(1.0f, 0, 0);
        original.set(2.0f, 0, 1);
        original.set(3.0f, 0, 2);
        assertEquals(3.4076058864593506f, VectorTensorMathUtils.logSumExp(original), 0.000001);
    }

    @Test
    public void percentileTest(){
        int rows =  1;
        int cols = 10;
        AbstractTensor original = new FloatBufferTensor(rows, cols);
        original.set(1.0f, 0, 0);
        original.set(2.0f, 0, 1);
        original.set(3.0f, 0, 2);
        original.set(-3.0f, 0, 3);
        original.set(3.0f, 0, 4);
        var x = VectorTensorMathUtils.valueBuckets(original);
        assertEquals(-3.0f, x.firstKey());
        assertEquals( "[2, 4]", x.get(3.0f).toString());
        assertEquals("{-3.0=[3], 0.0=[5, 6, 7, 8, 9], 1.0=[0], 2.0=[1], 3.0=[2, 4]}", x.toString());
        assertEquals(1, VectorTensorMathUtils.percentile(x, .90f, original.size()));
    }
}
