package io.teknek.deliverance.tensor;

import io.teknek.deliverance.DType;

import java.util.Objects;

public class ShapeKey {
    final TensorShape shape;
    final DType dType;

    ShapeKey(DType dType, TensorShape shape) {
        this.dType = dType;
        this.shape = shape;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ShapeKey shapeKey = (ShapeKey) o;
        return Objects.equals(shape, shapeKey.shape) && dType == shapeKey.dType;
    }

    @Override
    public int hashCode() {
        return Objects.hash(shape, dType);
    }
}
