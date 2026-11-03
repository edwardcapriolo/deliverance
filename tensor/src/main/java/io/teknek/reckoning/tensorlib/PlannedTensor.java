package io.teknek.reckoning.tensorlib;

import io.teknek.reckoning.tensor.AbstractTensor;

/** A materialized tensor paired with the TensorPlan node that last produced or touched it. */
public record PlannedTensor(AbstractTensor tensor, TensorPlan.Tensor plan) {
}
