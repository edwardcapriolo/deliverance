package io.teknek.deliverance.tensor.operations;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.tensor.*;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class PanamaTensorOperationsTest {

    static AbstractTensor allOnes(int size) {
        AbstractTensor f = new FloatBufferTensor(1, size);
        for (int i = 0; i < size; i++) {
            f.set(1.0f, 0, i);
        }
        return f;
    }

    static AbstractTensor random(int size, int seed){
        Random r = new Random(seed);
        AbstractTensor f = new FloatBufferTensor(1, size);
        for (int i=0;i< size;i++){
            f.set(r.nextFloat(-1,100), 0, i);
        }
        return f;
    }

    @Test
    void simpleDotProduct(){
        int size = 1024;
        NaiveTensorOperations controlOps = new NaiveTensorOperations();
        AbstractTensor a = allOnes(size);
        AbstractTensor b = allOnes(size);
        float control = controlOps.dotProduct(a, b, size);
        assertEquals(control, 1024f);

        PanamaTensorOperations p = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE, new TensorCache(new MetricRegistry()));
        assertEquals(control, p.dotProduct(a, b,1024));
    }


    @Test
    void dotProductQuantizationTest(){
        int size = 1024;
        int seed = 43;
        NaiveTensorOperations controlOps = new NaiveTensorOperations();
        AbstractTensor a = random(size, seed);
        AbstractTensor b = random(size, seed +1);
        AbstractTensor q8 = new Q8ByteBufferTensor(a);
        AbstractTensor q4 = new Q4ByteBufferTensor(b);

        float expected = 2587953.2f;
        float control = controlOps.dotProduct(q8, q4, size);
        assertEquals(expected, control);

        PanamaTensorOperations p = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE, new TensorCache(new MetricRegistry()));
        assertEquals(control, p.dotProduct(q8, q4, size), control * .01f);
    }
}
