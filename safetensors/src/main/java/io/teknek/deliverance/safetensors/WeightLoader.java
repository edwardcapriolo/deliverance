package io.teknek.deliverance.safetensors;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorInfo;

import java.util.Map;

public interface WeightLoader extends AutoCloseable {

    Map<String, String> metadata();

    Map<String, TensorInfo> tensorInfoMap();

    default boolean isWeightPresent(String name) {
        return tensorInfoMap().containsKey(name);
    }

    default AbstractTensor load(String name) {
        return load(name, null, false, false);
    }

    default AbstractTensor loadRows(String name, int rowOffset, int rowCount) {
        throw new UnsupportedOperationException("Row slicing not supported for " + getClass().getName());
    }

    AbstractTensor load(String name, DistributedContext dctx, boolean sparseRows, boolean sparseColumns);

    DType getModelDType();
}
