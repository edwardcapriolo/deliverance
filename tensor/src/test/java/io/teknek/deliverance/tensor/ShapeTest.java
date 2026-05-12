package io.teknek.deliverance.tensor;

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

    @Test
    void sparseRowExampleUsesLogicalCoordinates() {
        TensorShape s = TensorShape.sparseRow(new int[]{8, 6}, SparseOffset.of(2, 3));

        assertEquals(2, s.sparseRowOffset());
        assertEquals(3, s.sparseRowLength());
        assertEquals(6, s.sparseColumnLength());
        assertEquals(18, s.size());

        // Logical row 2 is the first resident row.
        assertEquals(0, s.getOffset(2, 0));
        assertEquals(5, s.getOffset(2, 5));

        // Logical row 4 is the third resident row.
        assertEquals(12, s.getOffset(4, 0));
        assertEquals(17, s.getOffset(4, 5));
    }

    @Test
    void sparseColumnExampleUsesLogicalCoordinates() {
        TensorShape s = TensorShape.sparseColumn(new int[]{3, 10}, SparseOffset.of(4, 3));

        assertEquals(4, s.sparseColumnOffset());
        assertEquals(3, s.sparseColumnLength());
        assertEquals(3, s.sparseRowLength());
        assertEquals(9, s.size());

        // Logical column 4 is the first resident column.
        assertEquals(0, s.getOffset(0, 4));
        assertEquals(2, s.getOffset(0, 6));

        // Logical column 4 in the next row starts after one resident row width.
        assertEquals(3, s.getOffset(1, 4));
        assertEquals(8, s.getOffset(2, 6));
    }

    @Test
    void sparseWindowsDoNotBoundsCheckResidentRange() {
        TensorShape sparseRows = TensorShape.sparseRow(new int[]{8, 6}, SparseOffset.of(2, 3));
        TensorShape sparseColumns = TensorShape.sparseColumn(new int[]{3, 10}, SparseOffset.of(4, 3));

        // Outside the resident row window but still inside the logical shape.
        assertEquals(-6, sparseRows.getOffset(1, 0));
        assertEquals(18, sparseRows.getOffset(5, 0));

        // Outside the resident column window but still inside the logical shape.
        assertEquals(-1, sparseColumns.getOffset(0, 3));
        assertEquals(3, sparseColumns.getOffset(0, 7));
    }

    @Test
    void scaleLastDimScalesDenseShape() {
        TensorShape s = TensorShape.of(4, 32);
        TensorShape scaled = s.scaleLastDim(0.5f);

        assertArrayEquals(new int[]{4, 16}, scaled.shapeArray());
        assertEquals(16, scaled.last());
        assertEquals(64, scaled.size());
    }

    @Test
    void scaleLastDimScalesSparseColumnWindow() {
        TensorShape s = TensorShape.sparseColumn(new int[]{3, 32}, SparseOffset.of(8, 16));
        TensorShape scaled = s.scaleLastDim(0.5f);

        assertArrayEquals(new int[]{3, 16}, scaled.shapeArray());
        assertEquals(4, scaled.sparseColumnOffset());
        assertEquals(8, scaled.sparseColumnLength());
        assertEquals(24, scaled.size());

        // Logical columns 4..11 are now the resident window in the scaled view.
        assertEquals(0, scaled.getOffset(0, 4));
        assertEquals(7, scaled.getOffset(0, 11));
        assertEquals(8, scaled.getOffset(1, 4));
    }

}
