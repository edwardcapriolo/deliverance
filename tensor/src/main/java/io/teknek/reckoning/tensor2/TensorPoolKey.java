package io.teknek.reckoning.tensor2;

import io.teknek.reckoning.DType;
import io.teknek.reckoning.tensor.TensorShape;

import java.util.Objects;

record TensorPoolKey(DType dType, TensorShape shape, String device) {
    TensorPoolKey {
        Objects.requireNonNull(dType, "dType");
        Objects.requireNonNull(shape, "shape");
        Objects.requireNonNull(device, "device");
    }
}
