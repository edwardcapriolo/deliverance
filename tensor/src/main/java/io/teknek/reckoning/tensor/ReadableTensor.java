package io.teknek.reckoning.tensor;

import io.teknek.reckoning.DType;

public interface ReadableTensor {
    TensorShape shape();
    DType dType();
    float get(int... dims);
    float get(int row, int column);
    long size();
}
