package io.teknek.deliverance.tensor;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class ShapeTest {

    @Test
    void oneDShape(){
        TensorShape s = TensorShape.ONE;
        assertArrayEquals(new int [] {1, 1}, s.shapeArray());

    }

    @Test
    void sliceTheSmallest(){
        TensorShape s = TensorShape.ONE;
        TensorShape k = s.slice(1);
        //We could consider throwing an exception here
        assertArrayEquals(new int [] {1, 1},  k.shapeArray());
    }

    @Test
    void slice2Dshape(){
        TensorShape s = TensorShape.of(2, 2);
        TensorShape k = s.slice(1);
        //If you slice a 2d array you get back a 2D array because we cant have 1D
        assertArrayEquals(new int [] {1, 2},  k.shapeArray());
    }

    @Test
    void slice3Dshape(){
        TensorShape s = TensorShape.of(2, 2, 2);
        TensorShape k = s.slice(1);
        //a 3d array when sliced becomes 2D
        assertArrayEquals(new int [] {2, 2}, k.shapeArray());
    }

    @Test
    void getOffset(){

        TensorShape s = TensorShape.of(2, 2, 2);
        assertEquals(4, s.getOffset(1, 0, 0));
        //Note get offset will gladly give offsets beyond the bounds of the tensor
        assertEquals(63, s.getOffset(9, 9, 9));

    }

    @Test
    void helpers(){
        TensorShape s = TensorShape.of(2, 3, 4);
        assertEquals(2, s.first());
        assertEquals(4, s.last());
    }

}
