package io.teknek.deliverance.tensor;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import org.junit.jupiter.api.Test;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;

public class FloatBufferTensorTest {

    @Test
    void basicTest(){
        try (FloatBufferTensor f = new FloatBufferTensor(10, 10)) {
            assertEquals(0,  f.get(1, 2));
            f.set(1.5f, 1 , 2);
            assertEquals(1.5f, f.get(1, 2));
            assertEquals(2, f.shape.dims());
            assertEquals(10, f.shape.dim(1));
            assertEquals(10, f.shape.dim(0));
            assertEquals(100, f.shape.size());
            assertThrowsExactly(IllegalArgumentException.class, () -> f.shape.dim(2));
            FloatVector fv = f.getVector(VectorSpecies.ofPreferred(Float.TYPE),0);
            FloatVector abs = fv.abs();
        }
    }

    @Test
    void floatVectorTest(){
        try (FloatBufferTensor f = new FloatBufferTensor(10, 10)) {
            f.set(1.5f, 0 , 0);
            f.set(-1.5f, 0 , 1);
            FloatVector fv = f.getVector(VectorSpecies.ofPreferred(Float.TYPE),0);
            FloatVector abs = fv.abs();
            float [] x = new float[100];
            abs.intoArray(x, 0);
            assertEquals(1.5, x[0], 0.001);
            assertEquals(1.5, x[1], 0.001);
        }
    }

    @Test
    void intoLargerArray(){
        try (FloatBufferTensor f = new FloatBufferTensor(10, 10)) {
            f.set(1.5f, 1, 0);
            f.set(-1.5f, 1, 1);
            FloatVector fv = f.getVector(VectorSpecies.ofPreferred(Float.TYPE), 1);
            FloatVector abs = fv.abs();
            float[] x = new float[100];
            abs.intoArray(x, 0);
            assertEquals(1.5, x[0], 0.001);
            assertEquals(1.5, x[0], 0.001);
        }
    }



    @Test
    void dubiousOffset(){
        try (FloatBufferTensor f = new FloatBufferTensor(10, 10)) {
            assertEquals(0, f.shape.getOffset(0));
            assertEquals(0, f.shape.getOffset(0, 0));
            // it doesnt throw when you go out of bounds should change this
            assertEquals(11, f.shape.getOffset(0, 11));
        }
    }

    @Test
    void sparsePropertiesTest(){
        try (FloatBufferTensor f = new FloatBufferTensor(10, 10)) {
            assertEquals(0, f.shape.getOffset(0));
            assertEquals(0, f.shape.getOffset(0, 0));
            // it doesnt throw when you go out of bounds should change this
            assertEquals(11, f.shape.getOffset(0, 11));
            assertFalse(f.shape().isSparse());
            assertEquals(10, f.shape().sparseRowLength());
            assertEquals(10, f.shape().sparseColumnLength());
        }
    }

    @Test
    void safeOffsetTest(){
        try (FloatBufferTensor f = new FloatBufferTensor(10, 10)) {
            assertEquals(Optional.empty(), f.shape.safeOffset(10, 10));
            assertEquals(Optional.of(0), f.shape.safeOffset(0, 0));
            assertEquals(Optional.of(99), f.shape.safeOffset(9, 9));
            assertEquals(Optional.empty(), f.shape.safeOffset(10, 1));
            assertEquals(Optional.empty(), f.shape.safeOffset(1, 10));
            assertEquals(Optional.empty(), f.shape.safeOffset(-1, 1));
        }
    }

}
