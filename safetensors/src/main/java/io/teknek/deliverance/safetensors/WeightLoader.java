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
        throw new UnsupportedOperationException("Weight loading not supported for " + getClass().getName());
    }

    default AbstractTensor loadRows(String name, int rowOffset, int rowCount) {
        throw new UnsupportedOperationException("Row slicing not supported for " + getClass().getName());
    }

    default AbstractTensor load(String name, TensorShardSpec shardSpec) {
        throw new UnsupportedOperationException("Tensor sharding not supported for " + getClass().getName());
    }

    DType getModelDType();
}
