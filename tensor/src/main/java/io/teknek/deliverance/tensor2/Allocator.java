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

    TensorRef allocate(DType dType, TensorShape shape) {
        return allocate(dType, shape, CPU);
    }

    TensorRef allocate(DType dType, TensorShape shape, String device) {
        Preconditions.checkArgument(CPU.equals(device), "Only cpu tensors are supported for now");
        if (dType == DType.I8 || dType == DType.Q4) {
            return allocateQuantized(dType, shape, device);
        }
        Preconditions.checkArgument(dType == DType.F32 || dType == DType.BF16,
                "Only F32, BF16, I8, and Q4 tensors are supported for now");
        return allocateDense(dType, shape, device, Map.of());
    }

    private TensorRef allocateDense(DType dType, TensorShape shape, String device, Map<String, TensorRef> sidecars) {
        TensorPoolKey key = new TensorPoolKey(dType, shape, device);
        Tensor tensor = availableByShape.computeIfAbsent(key, ignored -> new ConcurrentLinkedQueue<>()).poll();
        if (tensor == null) {
            tensor = newTensor(dType, shape);
        }
        return new TensorRef(new TensorRefState(LeaseState.USED, this, tensor, shape, dType, stride(shape), device,
                sidecars, null));
    }

    private TensorRef allocateQuantized(DType dType, TensorShape shape, String device) {
        String sidecarName = dType == DType.I8 ? Q8Layout.SCALE_SIDECAR : Q4Layout.SCALE_SIDECAR;
        TensorShape scaleShape = dType == DType.I8 ? Q8Layout.scaleShape(shape) : Q4Layout.scaleShape(shape);
        TensorRef scale = allocate(DType.F32, scaleShape, device);
        TensorRef tensor = allocateDense(dType, shape, device, Map.of(sidecarName, scale));
        if (dType == DType.I8) {
            ((I8Tensor) tensor.underlying()).attachScale(scale.underlying());
        } else {
            ((Q4Tensor) tensor.underlying()).attachScale(scale.underlying());
        }
        return tensor;
    }

    private Tensor newTensor(DType dType, TensorShape shape) {
        // TODO: evict or repurpose unused tensors from other shape pools when retained memory grows too large.
        return switch (dType) {
            case F32 -> new F32Tensor(shape);
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
