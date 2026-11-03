package io.teknek.reckoning.tensorlib;

import io.teknek.reckoning.tensor2.TensorRef;

/** A TensorRef materialization paired with its logical TensorPlan lineage. */
public record PlannedTensorRef(TensorRef tensor, TensorPlan.Tensor plan) implements AutoCloseable {
    @Override
    public void close() {
        tensor.close();
    }
}
