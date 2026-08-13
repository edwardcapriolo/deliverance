package io.teknek.deliverance.tensorlib;

import io.teknek.deliverance.tensor.AbstractTensor;

/** A materialized tensor paired with the TensorPlan node that last produced or touched it. */
public record PlannedTensor(AbstractTensor tensor, TensorPlan.Tensor plan) {
}
