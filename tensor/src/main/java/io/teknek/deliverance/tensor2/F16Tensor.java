package io.teknek.deliverance.tensor2;

import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.UnsafeDirectByteBuffer;

import java.lang.foreign.MemorySegment;
import java.nio.ShortBuffer;

final class F16Tensor extends Tensor {
    private final TensorShape shape;
    private final ShortBuffer underlyingByteBuffer;
    private final MemorySegment segment;

    F16Tensor(int... shape) {
        this(TensorShape.of(shape));
    }

    F16Tensor(TensorShape shape) {
        this.shape = shape;
        this.underlyingByteBuffer = UnsafeDirectByteBuffer.allocateAlignedByteBuffer(
                Ints.checkedCast(shape.size() * DType.F16.size()),
                UnsafeDirectByteBuffer.CACHE_LINE_SIZE
        ).asShortBuffer();
        this.segment = MemorySegment.ofBuffer(underlyingByteBuffer);
    }

    @Override
    public MemorySegment getMemorySegment() {
        return segment;
    }

    @Override
    public int getMemorySegmentOffset(int offset) {
        return offset * Short.BYTES;
    }

    @Override
    public float get(int... dims) {
        Preconditions.checkArgument(dims.length <= shape.dims(), "Too many dimensions specified");
        Preconditions.checkArgument(dims.length == shape.dims(), "Must specify all dimensions");
        return Float.float16ToFloat(underlyingByteBuffer.get(shape.getOffset(dims)));
    }

    @Override
    public float get(int row, int column) {
        Preconditions.checkArgument(shape.dims() == 2, "Shape is not 2 dimensions");
        return Float.float16ToFloat(underlyingByteBuffer.get(shape.getOffset(row, column)));
    }

    @Override
    public void set(float value, int row, int column) {
        Preconditions.checkArgument(shape.dims() == 2, "Shape is not 2 dimensions");
        underlyingByteBuffer.put(shape.getOffset(row, column), Float.floatToFloat16(value));
    }

    @Override
    public void set(float value, int... dims) {
        Preconditions.checkArgument(dims.length <= shape.dims(), "Too many dimensions specified");
        Preconditions.checkArgument(dims.length == shape.dims(), "Must specify all dimensions");
        underlyingByteBuffer.put(shape.getOffset(dims), Float.floatToFloat16(value));
    }
}
