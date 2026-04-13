package io.teknek.deliverance.tensor;

public interface ReadableTensor {
    float get(int... dims);
    long size();
}
