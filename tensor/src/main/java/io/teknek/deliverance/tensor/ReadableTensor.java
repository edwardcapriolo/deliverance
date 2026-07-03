package io.teknek.deliverance.tensor;

import io.teknek.deliverance.DType;

public interface ReadableTensor {
    TensorShape shape();
    DType dType();
    float get(int... dims);
    float get(int row, int column);
    long size();
}
