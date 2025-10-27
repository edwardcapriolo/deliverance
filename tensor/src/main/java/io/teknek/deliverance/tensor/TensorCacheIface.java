package io.teknek.deliverance.tensor;

import io.teknek.deliverance.DType;

public interface TensorCacheIface {
    AbstractTensor<?,?> getDirty(DType dType, TensorShape shape);
    AbstractTensor<?,?> get(DType dType, TensorShape shape);
    void release(AbstractTensor<?,?> b);
}
