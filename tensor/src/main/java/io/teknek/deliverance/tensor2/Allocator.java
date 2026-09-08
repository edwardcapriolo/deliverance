package io.teknek.deliverance.tensor2;

import com.google.common.base.Preconditions;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;

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
        Preconditions.checkArgument(dType == DType.F32 || dType == DType.BF16, "Only F32 and BF16 tensors are supported for now");
        Preconditions.checkArgument(CPU.equals(device), "Only cpu tensors are supported for now");
        TensorPoolKey key = new TensorPoolKey(dType, shape, device);
        Tensor tensor = availableByShape.computeIfAbsent(key, ignored -> new ConcurrentLinkedQueue<>()).poll();
        if (tensor == null) {
            // TODO: evict or repurpose unused tensors from other shape pools when retained memory grows too large.
            tensor = dType == DType.F32 ? new F32Tensor(shape) : new BF16Tensor(shape);
        }
        return new TensorRef(new TensorRefState(LeaseState.USED, this, tensor, shape, dType, stride(shape), device, null));
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
