package io.teknek.deliverance.tensor2;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.impl.Q4ByteBufferTensor;
import io.teknek.deliverance.tensor.impl.Q8ByteBufferTensor;
import io.teknek.dysfx.exception.UnreachableException;

import java.lang.foreign.MemorySegment;

import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

public class TensorRef implements AutoCloseable {

    private final AtomicReference<TensorRefState> state;

    TensorRef(TensorRefState state) {
        this.state = new AtomicReference<>(state);
    }

    public static TensorRef borrowed(AbstractTensor tensor) {
        Map<String, TensorRef> sidecars = switch (tensor) {
            case Q8ByteBufferTensor q8 -> Map.of(Q8Layout.SCALE_SIDECAR, borrowed(q8.getBlockF()));
            case Q4ByteBufferTensor q4 -> Map.of(Q4Layout.SCALE_SIDECAR, borrowed(q4.getBlockF()));
            default -> Map.of();
        };
        return borrowed(tensor, sidecars);
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

    private TensorRefState requireOpen() {
        TensorRefState current = state.get();
        if (current.leaseState() != LeaseState.USED) {
            throw new UnreachableException("Tensor is closed");
        }
        return current;
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
    }
}
