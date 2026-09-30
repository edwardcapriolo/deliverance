package io.teknek.deliverance.tensor2;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.dysfx.exception.UnreachableException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

class TensorRefSliceTest {

    @Test
    void sliceIsZeroCopyAndUsesLocalCoordinates() {
        Allocator allocator = new Allocator();
        TensorRef parent = allocator.allocate(DType.F32, TensorShape.of(3, 4));
        parent.underlying().set(7.0f, 1, 2);

        TensorRef row = parent.slice(1);

        assertEquals(TensorShape.of(1, 4), row.shape());
        assertEquals(7.0f, row.underlying().get(0, 2));
        assertSame(parent.memorySegment(), row.memorySegment());

        row.underlying().set(9.0f, 0, 2);
        assertEquals(9.0f, parent.underlying().get(1, 2));

        row.close();
        assertEquals(9.0f, parent.underlying().get(1, 2));
        parent.close();
    }

    @Test
    void parentCloseInvalidatesSlice() {
        Allocator allocator = new Allocator();
        TensorRef parent = allocator.allocate(DType.F32, TensorShape.of(3, 4));
        TensorRef row = parent.slice(1);

        parent.close();

        assertThrows(UnreachableException.class, row::shape);
        assertThrows(UnreachableException.class, () -> row.underlying().get(0, 0));
    }
}
