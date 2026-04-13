package io.teknek.deliverance.tensorlib;

import io.teknek.deliverance.tensor.ReadableTensor;

public interface ReadOnlyTensorMap<V> {
    V map(ReadableTensor t, long offset, long length);
}
