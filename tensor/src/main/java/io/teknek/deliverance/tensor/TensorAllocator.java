package io.teknek.deliverance.tensor;

import io.teknek.deliverance.DType;

public interface TensorAllocator {
    /**
     *
     * @return a tensor of the desired shape that has NOT been zero'ed out
     */
    AbstractTensor<?,?> getDirty(DType dType, TensorShape shape);
    /**
     *
     * @return a tensor of the desired shape with all elements set to 0
     * the ownerCache will be null.
     */
    AbstractTensor<?,?> get(DType dType, TensorShape shape);

    /**
     * Offer this tensor to an allocator so it can be reused
     * @param b a tensor no longer in use
     */
    void release(AbstractTensor<?,?> b);
}
