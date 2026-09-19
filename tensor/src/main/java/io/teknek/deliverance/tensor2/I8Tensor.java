package io.teknek.deliverance.tensor2;

import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.UnsafeDirectByteBuffer;

import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

class I8Tensor extends Tensor {
    private final TensorShape shape;
    private final ByteBuffer underlyingByteBuffer;
    private final MemorySegment segment;
    private Tensor scale;

    I8Tensor(TensorShape shape) {
        Preconditions.checkArgument(shape.last() % Q8Layout.BLOCK_SIZE == 0,
                "I8 tensor last dimension must be a multiple of %s", Q8Layout.BLOCK_SIZE);
        this.shape = shape;
        this.underlyingByteBuffer = UnsafeDirectByteBuffer.allocateAlignedByteBuffer(
                Ints.checkedCast(shape.size()),
                UnsafeDirectByteBuffer.CACHE_LINE_SIZE
        ).order(ByteOrder.LITTLE_ENDIAN);
        this.segment = MemorySegment.ofBuffer(underlyingByteBuffer);
    }

    void attachScale(Tensor scale) {
        this.scale = java.util.Objects.requireNonNull(scale, "scale");
    }

    @Override
    public float get(int... dims) {
        Preconditions.checkArgument(dims.length <= shape.dims(), "Too many dimensions specified");
        Preconditions.checkArgument(dims.length == shape.dims(), "Must specify all dimensions");
        Preconditions.checkState(scale != null, "I8 tensor scale sidecar is not attached");
        int[] scaleDims = java.util.Arrays.copyOf(dims, dims.length);
        scaleDims[scaleDims.length - 1] = Q8Layout.scaleColumn(scaleDims[scaleDims.length - 1]);
        return underlyingByteBuffer.get(shape.getOffset(dims)) * scale.get(scaleDims);
    }

    @Override
    public float get(int row, int column) {
        Preconditions.checkArgument(2 == shape.dims(), "Shape is not 2 dimensions");
        Preconditions.checkState(scale != null, "I8 tensor scale sidecar is not attached");
        return underlyingByteBuffer.get(shape.getOffset(row, column)) * scale.get(row, Q8Layout.scaleColumn(column));
    }

    @Override
    public void set(float v, int row, int column) {
        Preconditions.checkArgument(2 == shape.dims(), "Must specify all dimensions");
        Preconditions.checkState(scale != null, "I8 tensor scale sidecar is not attached");
        float factor = scale.get(row, Q8Layout.scaleColumn(column));
        underlyingByteBuffer.put(shape.getOffset(row, column), quantizedByte(v, factor));
    }

    @Override
    public void set(float v, int... dims) {
        Preconditions.checkArgument(dims.length <= shape.dims(), "Too many dimensions specified for tensor");
        Preconditions.checkArgument(dims.length == shape.dims(), "Must specify all dimensions");
        Preconditions.checkState(scale != null, "I8 tensor scale sidecar is not attached");
        int[] scaleDims = java.util.Arrays.copyOf(dims, dims.length);
        scaleDims[scaleDims.length - 1] = Q8Layout.scaleColumn(scaleDims[scaleDims.length - 1]);
        underlyingByteBuffer.put(shape.getOffset(dims), quantizedByte(v, scale.get(scaleDims)));
    }

    void setRawByte(byte value, int row, int column) {
        Preconditions.checkArgument(2 == shape.dims(), "Must specify all dimensions");
        underlyingByteBuffer.put(shape.getOffset(row, column), value);
    }

    byte getRawByte(int row, int column) {
        Preconditions.checkArgument(2 == shape.dims(), "Must specify all dimensions");
        return underlyingByteBuffer.get(shape.getOffset(row, column));
    }

    @Override
    public MemorySegment getMemorySegment() {
        return segment;
    }

    @Override
    public int getMemorySegmentOffset(int offset) {
        return offset;
    }

    private static byte quantizedByte(float value, float factor) {
        if (factor == 0.0f) {
            return 0;
        }
        int quantized = Math.round(value / factor);
        quantized = Math.max(Byte.MIN_VALUE, Math.min(Byte.MAX_VALUE, quantized));
        return (byte) quantized;
    }
}
