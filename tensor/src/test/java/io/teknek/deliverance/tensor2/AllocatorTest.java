package io.teknek.deliverance.tensor2;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.dysfx.exception.UnreachableException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

class AllocatorTest {

    @Test
    void allocatesOwnedCpuF32Tensor() {
        Allocator allocator = new Allocator();
        TensorShape shape = TensorShape.of(2, 3);

        TensorRef ref = allocator.allocate(DType.F32, shape);

        assertEquals(shape, ref.shape());
        assertEquals(DType.F32, ref.dType());
        assertEquals("cpu", ref.device());
        assertInstanceOf(F32Tensor.class, ref.underlying());
    }

    @Test
    void allocatesOwnedCpuF16Tensor() {
        Allocator allocator = new Allocator();
        TensorShape shape = TensorShape.of(2, 3);

        try (TensorRef ref = allocator.allocate(DType.F16, shape)) {
            assertEquals(shape, ref.shape());
            assertEquals(DType.F16, ref.dType());
            assertInstanceOf(F16Tensor.class, ref.underlying());
        }
    }

    @Test
    void closeReturnsTensorToPoolForShape() throws Exception {
        Allocator allocator = new Allocator();
        TensorShape shape = TensorShape.of(2, 3);
        TensorRef ref = allocator.allocate(DType.F32, shape);

        assertEquals(0, allocator.available(DType.F32, shape, "cpu"));

        ref.close();

        assertEquals(1, allocator.available(DType.F32, shape, "cpu"));
    }

    @Test
    void allocationReusesClosedTensorForSameShape() throws Exception {
        Allocator allocator = new Allocator();
        TensorShape shape = TensorShape.of(2, 3);
        TensorRef first = allocator.allocate(DType.F32, shape);
        Tensor firstTensor = first.underlying();

        first.close();
        TensorRef second = allocator.allocate(DType.F32, shape);

        assertSame(firstTensor, second.underlying());
        assertEquals(0, allocator.available(DType.F32, shape, "cpu"));
    }

    @Test
    void allocateZeroesReusedTensor() {
        Allocator allocator = new Allocator();
        TensorShape shape = TensorShape.of(1, 2);
        try (TensorRef first = allocator.allocate(DType.F32, shape)) {
            first.set(7.0f, 0, 0);
            first.set(-3.0f, 0, 1);
        }
        try (TensorRef second = allocator.allocate(DType.F32, shape)) {
            assertEquals(0.0f, second.get(0, 0));
            assertEquals(0.0f, second.get(0, 1));
        }
    }

    @Test
    void allocateDirtyReusesContentsByDefault() {
        Allocator allocator = new Allocator();
        TensorShape shape = TensorShape.of(1, 1);
        try (TensorRef first = allocator.allocate(DType.F32, shape)) {
            first.set(7.0f, 0, 0);
        }
        try (TensorRef second = allocator.allocateDirty(DType.F32, shape)) {
            assertEquals(7.0f, second.get(0, 0));
        }
    }

    @Test
    void disallowingDirtyAllocationsZeroesDirtyRequests() {
        Allocator allocator = new Allocator();
        allocator.allowDirtyTensors(false);
        TensorShape shape = TensorShape.of(1, 1);
        try (TensorRef first = allocator.allocate(DType.F32, shape)) {
            first.set(7.0f, 0, 0);
        }
        try (TensorRef second = allocator.allocateDirty(DType.F32, shape)) {
            assertEquals(0.0f, second.get(0, 0));
        }
    }

    @Test
    void closedTensorRefCannotBeUsed() throws Exception {
        Allocator allocator = new Allocator();
        TensorRef ref = allocator.allocate(DType.F32, TensorShape.of(2, 3));

        ref.close();

        assertThrows(UnreachableException.class, ref::shape);
    }

    @Test
    void rejectsUnsupportedDTypeAndDevice() {
        Allocator allocator = new Allocator();
        TensorShape shape = TensorShape.of(2, 3);

        assertThrows(IllegalArgumentException.class, () -> allocator.allocate(DType.I8, shape));
        assertThrows(IllegalArgumentException.class, () -> allocator.allocate(DType.F32, shape, "gpu"));
    }
}
