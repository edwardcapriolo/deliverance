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

    /**
     * Loads a local dense shard of a two-dimensional tensor.
     *
     * <p>The returned tensor uses local coordinates and local shape. For example, a column shard covering global columns
     * {@code [8, 16)} of a {@code [4, 32]} tensor is returned as a dense {@code [4, 8]} tensor, not as a sparse view with
     * the original global column indexes.</p>
     */
    default AbstractTensor load(String name, TensorShardSpec shardSpec) {
        throw new UnsupportedOperationException("Tensor sharding not supported for " + getClass().getName());
    }

    DType getModelDType();
}
