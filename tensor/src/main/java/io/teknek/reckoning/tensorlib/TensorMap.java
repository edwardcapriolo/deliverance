package io.teknek.reckoning.tensorlib;

import io.teknek.reckoning.tensor.AbstractTensor;

public interface TensorMap<V> {
    V map(AbstractTensor t, long start, long end);
}
