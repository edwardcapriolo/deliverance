package io.teknek.deliverance.tensor;

import io.teknek.deliverance.math.VectorMathUtils;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.api.Test;

import java.util.function.IntUnaryOperator;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

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

    @Test
    public void split(){

    }

    @Test
    public void softMaxRespectsOffsetWindow(){
        AbstractTensor original = new FloatBufferTensor(1, 5);
        original.set(99.0f, 0, 0);
        original.set(88.0f, 0, 1);
        original.set(1.0f, 0, 2);
        original.set(2.0f, 0, 3);
        original.set(3.0f, 0, 4);

        VectorTensorMathUtils.softMax(original, 2, 3);

        assertEquals(99.0f, original.get(0, 0), 0.000001);
        assertEquals(88.0f, original.get(0, 1), 0.000001);
        float sum = original.get(0, 2) + original.get(0, 3) + original.get(0, 4);
        assertEquals(1.0f, sum, 0.000001);
        assertTrue(original.get(0, 4) > original.get(0, 3));
        assertTrue(original.get(0, 3) > original.get(0, 2));
    }

    @Test
    public void scaledSoftMaxMatchesSeparateScaleAndSoftcapPasses() {
        AbstractTensor separate = new FloatBufferTensor(1, 5);
        AbstractTensor fused = new FloatBufferTensor(1, 5);
        for (int i = 0; i < 5; i++) {
            float value = (i - 2) * 1.75f;
            separate.set(value, 0, i);
            fused.set(value, 0, i);
        }

        float scale = 0.25f;
        float softcap = 1.5f;
        for (int i = 0; i < 5; i++) {
            float v = separate.get(0, i) * scale;
            v = (float) net.jafama.FastMath.tanh(v / softcap) * softcap;
            separate.set(v, 0, i);
        }
        VectorTensorMathUtils.softMax(separate, 0, 5);

        VectorTensorMathUtils.scaledSoftMax(fused, 0, 5, scale, softcap);

        for (int i = 0; i < 5; i++) {
            assertEquals(separate.get(0, i), fused.get(0, i), 0.000001, "col=" + i);
        }
    }

    @Test
    public void scaledSoftMaxMatchesSeparateScalePass() {
        AbstractTensor separate = new FloatBufferTensor(1, 5);
        AbstractTensor fused = new FloatBufferTensor(1, 5);
        for (int i = 0; i < 5; i++) {
            float value = (i - 2) * 1.25f;
            separate.set(value, 0, i);
            fused.set(value, 0, i);
        }

        float scale = 0.5f;
        for (int i = 0; i < 5; i++) {
            separate.set(separate.get(0, i) * scale, 0, i);
        }
        VectorTensorMathUtils.softMax(separate, 0, 5);

        VectorTensorMathUtils.scaledSoftMax(fused, 0, 5, scale, null);

        for (int i = 0; i < 5; i++) {
            assertEquals(separate.get(0, i), fused.get(0, i), 0.000001, "col=" + i);
        }
    }
}
