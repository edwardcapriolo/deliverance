package io.teknek.deliverance.model.tensorparallel.transport;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;

/**
 * Single-tensor binary payload codec inspired by safetensors' explicit dtype/shape/raw-bytes model.
 */
public class BinaryTensorPayloadCodec implements TensorPayloadCodec {
    private static final int MAGIC = 0x4454454E; // DTEN
    private static final int VERSION = 1;

    @Override
    public String contentType() {
        return "application/octet-stream";
    }

    @Override
    public byte[] encode(AbstractTensor tensor) {
        if (tensor.dType() != DType.F32) {
            throw new UnsupportedOperationException("Binary tensor payload currently supports F32 tensors");
        }
        int[] shape = tensor.shape().shapeArray();
        int byteLength = Math.toIntExact(tensor.size() * tensor.dType().size());
        ByteBuffer buffer = ByteBuffer.allocate(Integer.BYTES * 4 + Integer.BYTES * shape.length + Long.BYTES + byteLength)
                .order(ByteOrder.LITTLE_ENDIAN);
        buffer.putInt(MAGIC);
        buffer.putInt(VERSION);
        buffer.putInt(tensor.dType().ordinal());
        buffer.putInt(shape.length);
        for (int dim : shape) {
            buffer.putInt(dim);
        }
        buffer.putLong(byteLength);
        buffer.put(tensor.getMemorySegment().asByteBuffer().duplicate().order(ByteOrder.LITTLE_ENDIAN).slice(0, byteLength));
        return buffer.array();
    }

    @Override
    public AbstractTensor decode(byte[] bytes) {
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        int magic = buffer.getInt();
        if (magic != MAGIC) {
            throw new IllegalArgumentException("Invalid tensor payload magic");
        }
        int version = buffer.getInt();
        if (version != VERSION) {
            throw new IllegalArgumentException("Unsupported tensor payload version " + version);
        }
        DType dType = DType.values()[buffer.getInt()];
        if (dType != DType.F32) {
            throw new UnsupportedOperationException("Binary tensor payload currently supports F32 tensors");
        }
        int dims = buffer.getInt();
        if (dims < 1) {
            throw new IllegalArgumentException("Tensor payload must have at least one dimension");
        }
        int[] shape = new int[dims];
        long elements = 1;
        for (int i = 0; i < dims; i++) {
            shape[i] = buffer.getInt();
            if (shape[i] < 1) {
                throw new IllegalArgumentException("Tensor payload dimensions must be positive: " + Arrays.toString(shape));
            }
            elements *= shape[i];
        }
        long byteLength = buffer.getLong();
        long expectedLength = elements * dType.size();
        if (byteLength != expectedLength || byteLength != buffer.remaining()) {
            throw new IllegalArgumentException("Tensor payload byte length mismatch");
        }
        FloatBufferTensor tensor = new FloatBufferTensor(shape);
        tensor.getMemorySegment().asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).put(buffer.slice());
        return tensor;
    }
}
