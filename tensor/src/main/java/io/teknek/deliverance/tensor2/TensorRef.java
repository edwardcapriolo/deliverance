package io.teknek.deliverance.tensor2;

import com.google.common.base.Preconditions;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.impl.Q4ByteBufferTensor;
import io.teknek.deliverance.tensor.impl.Q8ByteBufferTensor;
import io.teknek.dysfx.exception.UnreachableException;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;

public class TensorRef implements AutoCloseable {

    private final AtomicReference<TensorRefState> state;
    private final TensorRef parent;
    private final Runnable closeAction;

    TensorRef(TensorRefState state) {
        this(state, null, () -> {
        });
    }

    private TensorRef(TensorRefState state, TensorRef parent) {
        this(state, parent, () -> {
        });
    }

    private TensorRef(TensorRefState state, TensorRef parent, Runnable closeAction) {
        this.state = new AtomicReference<>(state);
        this.parent = parent;
        this.closeAction = closeAction;
    }

    public static TensorRef borrowed(AbstractTensor tensor) {
        Map<String, TensorRef> sidecars = switch (tensor) {
            case Q8ByteBufferTensor q8 -> Map.of(Q8Layout.SCALE_SIDECAR, borrowed(q8.getBlockF()));
            case Q4ByteBufferTensor q4 -> Map.of(Q4Layout.SCALE_SIDECAR, borrowed(q4.getBlockF()));
            default -> Map.of();
        };
        return borrowed(tensor, sidecars);
    }

    /** Wraps an existing tensor and releases it when this ref is closed. */
    public static TensorRef owned(AbstractTensor tensor) {
        Map<String, TensorRef> sidecars = switch (tensor) {
            case Q8ByteBufferTensor q8 -> Map.of(Q8Layout.SCALE_SIDECAR, borrowed(q8.getBlockF()));
            case Q4ByteBufferTensor q4 -> Map.of(Q4Layout.SCALE_SIDECAR, borrowed(q4.getBlockF()));
            default -> Map.of();
        };
        TensorRef ref = borrowed(tensor, sidecars);
        return new TensorRef(ref.state.get(), null, tensor::close);
    }

    /** Creates a read-only TensorRef directly over mapped safetensor storage. */
    public static TensorRef mapped(MemorySegment payload, TensorShape shape, DType dType, int stride,
            Map<String, TensorRef> sidecars) {
        return mapped(payload, 0, payload.byteSize(), shape, dType, stride, sidecars);
    }

    public static TensorRef mapped(MemorySegment payload, long payloadOffset, long payloadLength,
            TensorShape shape, DType dType, int stride, Map<String, TensorRef> sidecars) {
        Objects.requireNonNull(payload, "payload");
        Objects.requireNonNull(shape, "shape");
        Objects.requireNonNull(dType, "dType");
        return new TensorRef(new TensorRefState(LeaseState.USED, null,
                new MappedTensor(payload, payloadOffset, payloadLength, shape, dType, sidecars), shape, dType, stride,
                        "cpu", sidecars, null));
    }

    /** Creates a read-only TensorRef over an existing mapped safetensor buffer. */
    public static TensorRef mapped(ByteBuffer payload, TensorShape shape, DType dType, int stride,
            Map<String, TensorRef> sidecars) {
        return mapped(payload, 0, payload.remaining(), shape, dType, stride, sidecars);
    }

    public static TensorRef mapped(ByteBuffer payload, int payloadOffset, int payloadLength,
            TensorShape shape, DType dType, int stride, Map<String, TensorRef> sidecars) {
        Objects.requireNonNull(payload, "payload");
        Objects.requireNonNull(shape, "shape");
        Objects.requireNonNull(dType, "dType");
        return new TensorRef(new TensorRefState(LeaseState.USED, null,
                new MappedTensor(payload, payloadOffset, payloadLength, shape, dType, sidecars), shape, dType, stride,
                        "cpu", sidecars, null));
    }

    private static TensorRef borrowed(AbstractTensor tensor, Map<String, TensorRef> sidecars) {
        return new TensorRef(new TensorRefState(LeaseState.USED, null, new Tensor() {
            @Override
            public float get(int... dims) {
                return tensor.get(dims);
            }

            @Override
            public float get(int row, int column) {
                return tensor.get(row, column);
            }

            @Override
            public void set(float v, int row, int column) {
                tensor.set(v, row, column);
            }

            @Override
            public void set(float v, int... dims) {
                tensor.set(v, dims);
            }

            @Override
            public MemorySegment getMemorySegment() {
                return tensor.getMemorySegment();
            }

            @Override
            public int getMemorySegmentOffset(int offset) {
                return tensor.getMemorySegmentOffset(offset);
            }
        }, tensor.shape(), tensor.dType(), tensor.getStride(), "cpu", sidecars, null));
    }

    private static final class MappedTensor extends Tensor {
        private final ByteBuffer payload;
        private final int payloadOffset;
        private final MemorySegment segment;
        private final long segmentOffset;
        private final MemorySegment memorySegment;
        private final TensorShape shape;
        private final DType dType;
        private final TensorRef scaleSidecar;

        private MappedTensor(MemorySegment payload, TensorShape shape, DType dType) {
            this(payload, 0, payload.byteSize(), shape, dType, Map.of());
        }

        private MappedTensor(MemorySegment payload, long payloadOffset, long payloadLength,
                TensorShape shape, DType dType, Map<String, TensorRef> sidecars) {
            this.payload = null;
            this.payloadOffset = 0;
            this.segment = payload;
            this.segmentOffset = payloadOffset;
            this.memorySegment = payload.asSlice(payloadOffset, payloadLength);
            this.shape = shape;
            this.dType = dType;
            this.scaleSidecar = sidecars.get(dType == DType.I8
                    ? Q8Layout.SCALE_SIDECAR : Q4Layout.SCALE_SIDECAR);
        }

        private MappedTensor(ByteBuffer payload, TensorShape shape, DType dType) {
            this(payload, 0, payload.remaining(), shape, dType, Map.of());
        }

        private MappedTensor(ByteBuffer payload, int payloadOffset, int payloadLength,
                TensorShape shape, DType dType, Map<String, TensorRef> sidecars) {
            this.payload = payload.order(ByteOrder.LITTLE_ENDIAN);
            this.payloadOffset = payloadOffset;
            this.segment = null;
            this.segmentOffset = 0;
            this.memorySegment = MemorySegment.ofBuffer(payload).asSlice(payloadOffset, payloadLength);
            this.shape = shape;
            this.dType = dType;
            this.scaleSidecar = sidecars.get(dType == DType.I8
                    ? Q8Layout.SCALE_SIDECAR : Q4Layout.SCALE_SIDECAR);
        }

        @Override
        public float get(int... dims) {
            return getAt(shape.getOffset(dims));
        }

        @Override
        public float get(int row, int column) {
            return getAt(shape.getOffset(row, column));
        }

        private float getAt(int offset) {
            long byteOffset = switch (dType) {
                case Q4 -> q4ByteOffset(offset);
                case F32 -> offset * 4L;
                default -> offset * (long) dType.size();
            };
            if (payload != null) {
                return switch (dType) {
                    case F32 -> payload.getFloat(Math.toIntExact(payloadOffset + byteOffset));
                    case BF16 -> io.teknek.deliverance.math.FloatConversions.bFloat16ToFloat32(
                            payload.getShort(Math.toIntExact(payloadOffset + byteOffset)));
                    case F16 -> Float.float16ToFloat(
                            payload.getShort(Math.toIntExact(payloadOffset + byteOffset)));
                    case I8 -> i8Value(offset, payload.get(Math.toIntExact(payloadOffset + byteOffset)));
                    case Q4 -> q4Value(offset, payload.get(Math.toIntExact(payloadOffset + byteOffset)));
                    default -> throw new UnsupportedOperationException("Mapped scalar reads do not support " + dType);
                };
            }
            return switch (dType) {
                case F32 -> segment.get(ValueLayout.JAVA_FLOAT_UNALIGNED, segmentOffset + byteOffset);
                case BF16 -> io.teknek.deliverance.math.FloatConversions.bFloat16ToFloat32(
                        segment.get(ValueLayout.JAVA_SHORT_UNALIGNED, segmentOffset + byteOffset));
                case F16 -> Float.float16ToFloat(
                        segment.get(ValueLayout.JAVA_SHORT_UNALIGNED, segmentOffset + byteOffset));
                case I8 -> i8Value(offset, segment.get(ValueLayout.JAVA_BYTE, segmentOffset + byteOffset));
                case Q4 -> q4Value(offset, segment.get(ValueLayout.JAVA_BYTE, segmentOffset + byteOffset));
                default -> throw new UnsupportedOperationException("Mapped scalar reads do not support " + dType);
            };
        }

        private float q4Value(int offset, byte packed) {
            if (scaleSidecar == null) {
                throw new UnsupportedOperationException("Scalar Q4 reads require a Q4 scale sidecar");
            }
            int nibble = offset % Q4Layout.BLOCK_SIZE < Q4Layout.HALF_BLOCK
                    ? packed & 0x0F : (packed >>> 4) & 0x0F;
            int row = shape.dims() == 2 ? offset / shape.last() : 0;
            int column = shape.dims() == 2 ? offset % shape.last() : offset;
            return (nibble - 8) * scaleSidecar.get(row, column / Q4Layout.BLOCK_SIZE);
        }

        private float i8Value(int offset, byte value) {
            if (scaleSidecar == null) {
                throw new UnsupportedOperationException("Scalar I8 reads require an I8 scale sidecar");
            }
            int row = shape.dims() == 2 ? offset / shape.last() : 0;
            int column = shape.dims() == 2 ? offset % shape.last() : offset;
            return value * scaleSidecar.get(row, column / Q8Layout.BLOCK_SIZE);
        }

        private static int q4ByteOffset(int offset) {
            return (offset / Q4Layout.BLOCK_SIZE) * Q4Layout.HALF_BLOCK
                    + offset % Q4Layout.HALF_BLOCK;
        }

        @Override
        public void set(float value, int row, int column) {
            throw new UnsupportedOperationException("Mapped model weights are read-only");
        }

        @Override
        public void set(float value, int... dims) {
            throw new UnsupportedOperationException("Mapped model weights are read-only");
        }

        @Override
        public MemorySegment getMemorySegment() {
            return memorySegment;
        }

        @Override
        public int getMemorySegmentOffset(int offset) {
            return dType == DType.Q4 ? offset / 2 : offset * dType.size();
        }
    }

    private TensorRefState requireOpen() {
        TensorRefState current = state.get();
        if (current.leaseState() != LeaseState.USED) {
            throw new UnreachableException("Tensor is closed");
        }
        if (parent != null) {
            parent.requireOpen();
        }
        return current;
    }

    /** Returns a zero-copy borrowed view tied to this ref's lifetime. */
    public TensorRef slice(int... dims) {
        TensorRefState source = requireOpen();
        Preconditions.checkArgument(dims.length < source.shape().dims(),
                "Too many dimensions specified for tensor");
        int[] sourceShape = source.shape().shapeArray();
        for (int i = 0; i < dims.length; i++) {
            Preconditions.checkArgument(dims[i] >= 0 && dims[i] < sourceShape[i],
                    "Slice dimension %s is out of bounds", dims[i]);
        }

        TensorShape childShape = source.shape().slice(dims.length);
        int[] baseCoordinates = java.util.Arrays.copyOf(dims, sourceShape.length);
        int baseOffset = source.shape().getOffset(baseCoordinates);
        Map<String, TensorRef> childSidecars = new java.util.HashMap<>();
        for (Map.Entry<String, TensorRef> sidecar : source.sidecars().entrySet()) {
            childSidecars.put(sidecar.getKey(), sidecar.getValue().slice(dims));
        }
        Tensor childTensor = new Tensor() {
            private int[] coordinates(int... localDims) {
                Preconditions.checkArgument(localDims.length == childShape.dims(),
                        "Expected %s dimensions", childShape.dims());
                int localStart = source.shape().dims() - dims.length == 1 && childShape.dims() == 2 ? 1 : 0;
                int[] full = new int[dims.length + localDims.length - localStart];
                System.arraycopy(dims, 0, full, 0, dims.length);
                System.arraycopy(localDims, localStart, full, dims.length, localDims.length - localStart);
                return full;
            }

            @Override
            public float get(int... localDims) {
                TensorRef.this.requireOpen();
                return source.underlying().get(coordinates(localDims));
            }

            @Override
            public float get(int row, int column) {
                TensorRef.this.requireOpen();
                return source.underlying().get(coordinates(row, column));
            }

            @Override
            public void set(float value, int row, int column) {
                TensorRef.this.requireOpen();
                source.underlying().set(value, coordinates(row, column));
            }

            @Override
            public void set(float value, int... localDims) {
                TensorRef.this.requireOpen();
                source.underlying().set(value, coordinates(localDims));
            }

            @Override
            public MemorySegment getMemorySegment() {
                TensorRef.this.requireOpen();
                return source.underlying().getMemorySegment();
            }

            @Override
            public int getMemorySegmentOffset(int offset) {
                TensorRef.this.requireOpen();
                return source.underlying().getMemorySegmentOffset(baseOffset + offset);
            }
        };
        TensorRefState childState = new TensorRefState(LeaseState.USED, null, childTensor, childShape,
                source.dType(), stride(childShape), source.device(), childSidecars, source);
        return new TensorRef(childState, this);
    }

    private static int stride(TensorShape shape) {
        return shape.first() > 1 && shape.dims() == 2 ? shape.getOffset(1, 0) : 0;
    }

    Tensor underlying() {
        return requireOpen().underlying();
    }

    public DType dType() {
        return requireOpen().dType();
    }

    String device() {
        return requireOpen().device();
    }

    public int stride() {
        return requireOpen().stride();
    }

    public TensorRef sidecar(String name) {
        return requireOpen().sidecars().get(name);
    }

    public MemorySegment memorySegment() {
        return underlying().getMemorySegment();
    }

    public int memorySegmentOffset(int offset) {
        return underlying().getMemorySegmentOffset(offset);
    }

    public TensorShape getShape(){
        return requireOpen().shape();
    }
    public TensorShape shape(){
        return requireOpen().shape();
    }

    public int dims() {
        return shape().dims();
    }

    public float get(int... dims) {
        return underlying().get(dims);
    }

    public float get(int row, int column) {
        return underlying().get(row, column);
    }

    public void set(float value, int... dims) {
        underlying().set(value, dims);
    }

    public void set(float value, int row, int column) {
        underlying().set(value, row, column);
    }

    /** The caller is done with the object and can be retuned to the pool */
    @Override
    public void close() {
        TensorRefState current = requireOpen();
        TensorRefState closed = TensorRefState.closed(current);
        if (!state.compareAndSet(current, closed)) {
            throw new UnreachableException("Double close on tensor");
        }
        current.sidecars().values().forEach(TensorRef::close);
        if (current.allocator() != null) {
            current.allocator().close(current);
        }
        closeAction.run();
    }
}
