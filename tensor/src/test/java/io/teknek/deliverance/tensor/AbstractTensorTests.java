package io.teknek.deliverance.tensor;

import io.teknek.deliverance.tensor.impl.FloatBufferTensor;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.*;

public class AbstractTensorTests {

    @Test
    public void oneArgumentToShape() {
        //given one argument to the shape of a tensor
        AbstractTensor x = new FloatBufferTensor(4);
        //then the number of dims is two
        assertEquals(2, x.dims());
        //and the shape is 1 row 4 columns
        assertArrayEquals(new int[]{1, 4}, x.shape().shapeArray());
        //and the size is 1 * 4
        assertEquals(4, x.size());
    }

    @Test
    public void twoArgumentToShape(){
        AbstractTensor f = new FloatBufferTensor(4, 8);
        assertEquals(2, f.dims());
        assertArrayEquals(new int[]{4, 8}, f.shape().shapeArray());
        assertEquals(32, f.size());
    }


    @Test
    void sliceTest(){
        int rows = 4;
        int columns = 8;
        AbstractTensor f = new FloatBufferTensor(rows, columns);
        for (int i = 0; i < rows * columns; i++) {
            f.set(i, 0, i);
        }
        AbstractTensor slided = f.slice( 1);
        assertArrayEquals(slided.shape.shapeArray(), new int[] { 1, columns });
        assertEquals(columns, slided.size());
        List<Float> results = new ArrayList<>();
        for (int i = 0; i < slided.size(); i++) {
            results.add(slided.get(0, i));
        }
        assertEquals(Arrays.asList(8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f), results);
    }

    /**
     * original
     * [0][0]=  0.0000 [0][1]=  1.0000 [0][2]=  2.0000 [0][3]=  3.0000 
     * [1][0]=  4.0000 [1][1]=  5.0000 [1][2]=  6.0000 [1][3]=  7.0000 
     * [2][0]=  8.0000 [2][1]=  9.0000 [2][2]= 10.0000 [2][3]= 11.0000 
     * original.slice(1);
     * [0][0]=  4.0000 [0][1]=  5.0000 [0][2]=  6.0000 [0][3]=  7.0000 
     * 
     */
    @Test
    public void updateToSliceChangesOriginal(){
        int rows = 3;
        int columns = 4;
        AbstractTensor original = new FloatBufferTensor(rows, columns);
        for (int i = 0; i < rows * columns; i++) {
            original.set(i, 0, i);
        }

        AbstractTensor sliced = original.slice(1);
        System.out.println(TensorDisplayUtil.pretty2dDisplayAll(original));
        System.out.println(TensorDisplayUtil.pretty2dDisplayAll(sliced));
        sliced.set(10.0f, 0, 0);
        System.out.println(TensorDisplayUtil.pretty2dDisplayAll(original));
        System.out.println(TensorDisplayUtil.pretty2dDisplayAll(sliced));
        assertEquals(10F, sliced.get(0, 0));
        assertEquals(10F, original.get(1, 0));
    }


    @Test
    void iterateTest(){
        int rows = 4;
        int columns = 8;
        AbstractTensor f = new FloatBufferTensor(rows, columns);
        for (int i = 0; i < rows * columns; i++) {
            f.set(i, 0, i);
        }
        int [] iterator = new int[]{0, 0};
        boolean x = f.iterate(iterator);
        assertTrue(x);
        assertEquals(Arrays.toString(iterator), Arrays.toString(new int[] {0, 1}));
        iterator[1]= 7;
        x = f.iterate(iterator);
        assertTrue(x);
        assertEquals(Arrays.toString(iterator), Arrays.toString(new int[] {1, 0}));

        //when it is at the end
        iterator[0] = 3;
        iterator[1] = 7;
        x = f.iterate(iterator);
        assertFalse(x);
        assertEquals(Arrays.toString(iterator), Arrays.toString(new int[] {0, 0}));
    }
    
    @Test
    void iterate3dTest(){
        int rows = 2;
        int columns = 3;
        int depth = 2;
        TensorShape shape = TensorShape.of(rows, columns, depth);
        Assertions.assertEquals( 12, shape.size());
        AbstractTensor f = new FloatBufferTensor(rows, columns, depth);
        Assertions.assertEquals( 12, f.size());
        int z = 0;
        for (int i = 0; i < rows; i++) {
            for (int j =0 ; j < columns; j++){
                for (int k = 0; k < depth; k++){
                    f.set(z++, i, j, k);
                }
            }
        }
        {
            //Wrong sized iterator throws
            int[] iterator = new int[]{0, 0};
            Assertions.assertThrows(IllegalArgumentException.class, () -> f.iterate(iterator));
        }
        int [] iterator = new int[]{0, 0, 0};
        List<Float> results = new ArrayList<>();
        List<String> indexes = new ArrayList<>();
        do {
            results.add(f.get(iterator));
            indexes.add(Arrays.toString(iterator));
        } while(f.iterate(iterator));
        assertEquals( "[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]", results.toString()  );
        assertEquals("[[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [0, 2, 0], [0, 2, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1], [1, 2, 0], [1, 2, 1]]", indexes.toString());

    }
}
