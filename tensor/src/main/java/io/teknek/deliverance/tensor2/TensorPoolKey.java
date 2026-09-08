package io.teknek.deliverance.tensor2;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;

import java.util.Objects;

record TensorPoolKey(DType dType, TensorShape shape, String device) {
    TensorPoolKey {
        Objects.requireNonNull(dType, "dType");
        Objects.requireNonNull(shape, "shape");
        Objects.requireNonNull(device, "device");
    }
}
