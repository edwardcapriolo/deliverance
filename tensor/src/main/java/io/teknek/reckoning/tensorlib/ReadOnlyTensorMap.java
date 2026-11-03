package io.teknek.reckoning.tensorlib;

import io.teknek.reckoning.tensor.ReadableTensor;

public interface ReadOnlyTensorMap<V> {
    V map(ReadableTensor t, long offset, long length);
}
