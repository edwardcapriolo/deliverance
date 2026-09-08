package io.teknek.deliverance.tensor2;

import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.UnsafeDirectByteBuffer;

import java.lang.foreign.MemorySegment;
import java.nio.FloatBuffer;

class F32Tensor extends Tensor {
    private final TensorShape shape;
    final FloatBuffer underlyingByteBuffer;
    final MemorySegment segment;

    F32Tensor(int... shape) {
        this(TensorShape.of(shape));
    }

    F32Tensor(TensorShape shape) {
        this.shape = shape;
        this.underlyingByteBuffer = UnsafeDirectByteBuffer.allocateAlignedByteBuffer(
                Ints.checkedCast(shape.size() * DType.F32.size()),
                UnsafeDirectByteBuffer.CACHE_LINE_SIZE
        ).asFloatBuffer();
        this.segment = MemorySegment.ofBuffer(underlyingByteBuffer);
    }

    public MemorySegment getMemorySegment() {
        return segment;
    }

    @Override
    public int getMemorySegmentOffset(int offset) {
        return offset * Float.BYTES;
    }

    @Override
    public float get(int... dims) {
        Preconditions.checkArgument(dims.length <= shape.dims(), "Too many dimensions specified");
        Preconditions.checkArgument(dims.length == shape.dims(), "Must specify all dimensions");
        return underlyingByteBuffer.get(shape.getOffset(dims));
    }

    @Override
    public float get(int row, int column) {
        Preconditions.checkArgument(2 == shape.dims(), "Shape is not 2 dimensions");
        return underlyingByteBuffer.get(shape.getOffset(row, column));
    }

    @Override
    public void set(float v, int row, int column) {
        Preconditions.checkArgument(2 == shape.dims(), "Must specify all dimensions");
        underlyingByteBuffer.put(shape.getOffset(row, column), v);
    }

    @Override
    public void set(float v, int... dims) {
        Preconditions.checkArgument(dims.length <= shape.dims(), "Too many dimensions specified for tensor");
        Preconditions.checkArgument(dims.length == shape.dims(), "Must specify all dimensions");
        underlyingByteBuffer.put(shape.getOffset(dims), v);
    }
}
