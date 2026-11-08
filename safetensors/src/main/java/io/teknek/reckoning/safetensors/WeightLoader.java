package io.teknek.reckoning.safetensors;

import io.teknek.reckoning.DType;
import io.teknek.reckoning.tensor.TensorInfo;
import io.teknek.reckoning.tensor2.Lighter;
import io.teknek.reckoning.tensor2.TensorRef;

import java.util.Map;
import java.nio.file.Path;
import java.util.Optional;

public interface WeightLoader extends AutoCloseable {

    Map<String, String> metadata();

    Map<String, TensorInfo> tensorInfoMap();

    default boolean isWeightPresent(String name) {
        return tensorInfoMap().containsKey(name);
    }

    default TensorRef load(String name) {
        throw new UnsupportedOperationException("Weight loading not supported for " + getClass().getName());
    }

    /**
     * Compatibility alias for callers that still spell the TensorRef load operation explicitly.
     */
    default TensorRef loadRef(String name) {
        return load(name);
    }

    default TensorRef loadRef(String name, TensorShardSpec shardSpec) {
        return load(name, shardSpec);
    }

    default TensorRef loadRows(String name, int rowOffset, int rowCount) {
        throw new UnsupportedOperationException("Row slicing not supported for " + getClass().getName());
    }

    /**
     * Loads a local dense shard of a two-dimensional tensor.
     *
     * <p>The returned tensor uses local coordinates and local shape. For example, a column shard covering global columns
     * {@code [8, 16)} of a {@code [4, 32]} tensor is returned as a dense {@code [4, 8]} tensor, not as a sparse view with
     * the original global column indexes.</p>
     */
    default TensorRef load(String name, TensorShardSpec shardSpec) {
        throw new UnsupportedOperationException("Tensor sharding not supported for " + getClass().getName());
    }

    DType getModelDType();

    /** Controls implicit majority-dtype inference used by legacy model loaders. */
    default void setUseMajorityDType(boolean useMajorityDType) {
    }

    /**
     * Supplies the model's tensor2 execution context before model initialization loads weights.
     * Legacy loaders that do not expose tensor2 weights may leave this as a no-op.
     */
    default void setLighter(Lighter lighter) {
    }

    default Optional<Path> modelRoot() {
        return Optional.empty();
    }
}
