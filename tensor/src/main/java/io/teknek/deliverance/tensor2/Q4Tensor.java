package io.teknek.deliverance.tensor2;

import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.UnsafeDirectByteBuffer;

import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

class Q4Tensor extends Tensor {
    private final TensorShape shape;
    private final ByteBuffer underlyingByteBuffer;
    private final MemorySegment segment;
    private Tensor scale;

    Q4Tensor(TensorShape shape) {
        Preconditions.checkArgument(shape.last() % Q4Layout.BLOCK_SIZE == 0,
                "Q4 tensor last dimension must be a multiple of %s", Q4Layout.BLOCK_SIZE);
        this.shape = shape;
        this.underlyingByteBuffer = UnsafeDirectByteBuffer.allocateAlignedByteBuffer(
                Ints.checkedCast(shape.size() / 2),
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
        Preconditions.checkState(scale != null, "Q4 tensor scale sidecar is not attached");
        int offset = shape.getOffset(dims);
        int[] scaleDims = java.util.Arrays.copyOf(dims, dims.length);
        scaleDims[scaleDims.length - 1] = Q4Layout.blockIndex(scaleDims[scaleDims.length - 1]);
        return signedNibble(offset) * scale.get(scaleDims);
    }

    @Override
    public float get(int row, int column) {
        Preconditions.checkArgument(2 == shape.dims(), "Shape is not 2 dimensions");
        Preconditions.checkState(scale != null, "Q4 tensor scale sidecar is not attached");
        int offset = shape.getOffset(row, column);
        return signedNibble(offset) * scale.get(row, Q4Layout.blockIndex(column));
    }

    @Override
    public void set(float v, int row, int column) {
        throw new UnsupportedOperationException("Q4 values must be written through quantization helpers");
    }

    @Override
    public void set(float v, int... dims) {
        throw new UnsupportedOperationException("Q4 values must be written through quantization helpers");
    }

    void setPackedByte(byte value, int row, int blockColumn, int packedColumn) {
        Preconditions.checkArgument(2 == shape.dims(), "Must specify all dimensions");
        Preconditions.checkArgument(packedColumn >= 0 && packedColumn < Q4Layout.HALF_BLOCK,
                "Packed Q4 column must be within a half block");
        int column = blockColumn * Q4Layout.BLOCK_SIZE + packedColumn;
        underlyingByteBuffer.put(byteIndex(shape.getOffset(row, column)), value);
    }

    byte getPackedByte(int row, int blockColumn, int packedColumn) {
        Preconditions.checkArgument(2 == shape.dims(), "Must specify all dimensions");
        Preconditions.checkArgument(packedColumn >= 0 && packedColumn < Q4Layout.HALF_BLOCK,
                "Packed Q4 column must be within a half block");
        int column = blockColumn * Q4Layout.BLOCK_SIZE + packedColumn;
        return underlyingByteBuffer.get(byteIndex(shape.getOffset(row, column)));
    }

    @Override
    public MemorySegment getMemorySegment() {
        return segment;
    }

    @Override
    public int getMemorySegmentOffset(int offset) {
        return offset / 2;
    }

    private int signedNibble(int offset) {
        int blockOffset = offset % Q4Layout.BLOCK_SIZE;
        byte packed = underlyingByteBuffer.get(byteIndex(offset));
        if (blockOffset < Q4Layout.HALF_BLOCK) {
            return (packed & 0x0F) - 8;
        }
        return ((packed >> 4) & 0x0F) - 8;
    }

    private static int byteIndex(int offset) {
        int blockOffset = offset % Q4Layout.BLOCK_SIZE;
        int base = Q4Layout.blockIndex(offset) * Q4Layout.HALF_BLOCK;
        return base + (blockOffset % Q4Layout.HALF_BLOCK);
    }
}
