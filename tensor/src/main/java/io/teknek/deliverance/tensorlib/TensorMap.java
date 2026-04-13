package io.teknek.deliverance.tensorlib;

import io.teknek.deliverance.tensor.AbstractTensor;

public interface TensorMap<V> {
    V map(AbstractTensor t, long start, long end);
}
