package io.teknek.deliverance.tensor;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.google.common.primitives.Ints;
import io.teknek.deliverance.DType;

import java.util.Arrays;
import java.util.Objects;

public class TensorInfo implements Comparable<TensorInfo> {

    @JsonProperty("dtype")
    public final DType dType;

    @JsonProperty("shape")
    public final int[] shape;

    @JsonProperty("data_offsets")
    public final long[] dataOffsets;

    @JsonCreator
    public TensorInfo(
            @JsonProperty("dtype") DType dType,
            @JsonProperty("shape") long[] shape,
            @JsonProperty("data_offsets") long[] dataOffsets
    ) {
        this.dType = dType;
        this.shape = new int[shape.length];
        for (int i = 0; i < shape.length; i++)
            this.shape[i] = Ints.checkedCast(shape[i]);
        this.dataOffsets = dataOffsets;
    }

    @Override
    public String toString() {
        return "TensorInfo{"
                + "dType="
                + dType
                + ", shape="
                + Arrays.toString(shape)
                + ", dataOffsets="
                + Arrays.toString(dataOffsets)
                + "}";
    }

    @Override
    public final boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof TensorInfo that)) return false;

        return dType == that.dType && Arrays.equals(shape, that.shape) && Arrays.equals(dataOffsets, that.dataOffsets);
    }

    @Override
    public int hashCode() {
        int result = Objects.hashCode(dType);
        result = 31 * result + Arrays.hashCode(shape);
        result = 31 * result + Arrays.hashCode(dataOffsets);
        return result;
    }

    @Override
    public int compareTo(TensorInfo o) {
        // In the case we are reading in order of dataOffsets
        return Long.compare(dataOffsets[0], o.dataOffsets[0]);
    }
}