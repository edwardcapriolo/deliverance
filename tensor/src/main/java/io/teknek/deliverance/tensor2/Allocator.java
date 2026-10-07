package io.teknek.deliverance.tensor2;

import com.google.common.base.Preconditions;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ConcurrentMap;

class Allocator {
    //This is be a TensorAllocator howwever to must unerstand multiple devices
    //it must always allocate the tensor its not a "cache" that hands back raw tensors when its "full"
    private static final String CPU = "cpu";
    private final ConcurrentMap<TensorPoolKey, ConcurrentLinkedQueue<Tensor>> availableByShape = new ConcurrentHashMap<>();
    private volatile boolean allowDirtyTensors = true;

    TensorRef allocate(DType dType, TensorShape shape) {
        return allocate(dType, shape, CPU, true);
    }

    TensorRef allocateDirty(DType dType, TensorShape shape) {
        return allocate(dType, shape, CPU, !allowDirtyTensors);
    }

    void allowDirtyTensors(boolean allowDirtyTensors) {
        this.allowDirtyTensors = allowDirtyTensors;
    }

    TensorRef allocate(DType dType, TensorShape shape, String device) {
        return allocate(dType, shape, device, true);
    }

    private TensorRef allocate(DType dType, TensorShape shape, String device, boolean zero) {
        Preconditions.checkArgument(CPU.equals(device), "Only cpu tensors are supported for now");
        if (dType == DType.I8 || dType == DType.Q4) {
            return allocateQuantized(dType, shape, device, zero);
        }
        Preconditions.checkArgument(dType == DType.F32 || dType == DType.F16 || dType == DType.BF16,
                "Only F32, F16, BF16, I8, and Q4 tensors are supported for now");
        return allocateDense(dType, shape, device, Map.of(), zero);
    }

    private TensorRef allocateDense(DType dType, TensorShape shape, String device, Map<String, TensorRef> sidecars,
            boolean zero) {
        TensorPoolKey key = new TensorPoolKey(dType, shape, device);
        Tensor tensor = availableByShape.computeIfAbsent(key, ignored -> new ConcurrentLinkedQueue<>()).poll();
        if (tensor == null) {
            tensor = newTensor(dType, shape);
        }
        TensorRef ref = new TensorRef(new TensorRefState(LeaseState.USED, this, tensor, shape, dType, stride(shape),
                device, sidecars, null));
        if (zero) {
            clear(ref);
        }
        return ref;
    }

    private TensorRef allocateQuantized(DType dType, TensorShape shape, String device, boolean zero) {
        String sidecarName = dType == DType.I8 ? Q8Layout.SCALE_SIDECAR : Q4Layout.SCALE_SIDECAR;
        TensorShape scaleShape = dType == DType.I8 ? Q8Layout.scaleShape(shape) : Q4Layout.scaleShape(shape);
        TensorRef scale = allocate(DType.F32, scaleShape, device, zero);
        TensorRef tensor = allocateDense(dType, shape, device, Map.of(sidecarName, scale), zero);
        if (dType == DType.I8) {
            ((I8Tensor) tensor.underlying()).attachScale(scale.underlying());
        } else {
            ((Q4Tensor) tensor.underlying()).attachScale(scale.underlying());
        }
        return tensor;
    }

    void clear(TensorRef target) {
        Tensor tensor = target.underlying();
        long bytes = target.dType() == DType.Q4
                ? target.shape().size() / 2
                : target.shape().size() * target.dType().size();
        byte zero = target.dType() == DType.Q4 ? (byte) 0x88 : 0;
        tensor.getMemorySegment().asSlice(tensor.getMemorySegmentOffset(0), bytes).fill(zero);
        if (target.dType() == DType.I8) {
            clear(target.sidecar(Q8Layout.SCALE_SIDECAR));
        } else if (target.dType() == DType.Q4) {
            clear(target.sidecar(Q4Layout.SCALE_SIDECAR));
        }
    }

    private Tensor newTensor(DType dType, TensorShape shape) {
        // TODO: evict or repurpose unused tensors from other shape pools when retained memory grows too large.
        return switch (dType) {
            case F32 -> new F32Tensor(shape);
            case F16 -> new F16Tensor(shape);
            case BF16 -> new BF16Tensor(shape);
            case I8 -> new I8Tensor(shape);
            case Q4 -> new Q4Tensor(shape);
            default -> throw new IllegalArgumentException("Unsupported tensor dtype " + dType);
        };
    }

    void close(TensorRefState state) {
        TensorPoolKey key = new TensorPoolKey(state.dType(), state.shape(), state.device());
        availableByShape.computeIfAbsent(key, ignored -> new ConcurrentLinkedQueue<>()).offer(state.underlying());
    }

    int available(DType dType, TensorShape shape, String device) {
        ConcurrentLinkedQueue<Tensor> queue = availableByShape.get(new TensorPoolKey(dType, shape, device));
        return queue == null ? 0 : queue.size();
    }

    private static int stride(TensorShape shape) {
        return shape.first() > 1 && shape.dims() == 2 ? shape.getOffset(shape.sparseRowOffset() + 1, shape.sparseColumnOffset()) : 0;
    }
}
