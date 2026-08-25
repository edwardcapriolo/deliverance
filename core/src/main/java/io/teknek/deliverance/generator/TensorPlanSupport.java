package io.teknek.deliverance.generator;

import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.tensor.operations.TensorOperations;
import io.teknek.deliverance.tensorlib.TensorPlan;

final class TensorPlanSupport {
    private TensorPlanSupport() {
    }

    static TensorPlan plan(AbstractModel model, TensorOperations operations) {
        return new TensorPlan(operations, model.getPool(), model.getMetricRegistry(), model, model.getTensorRuntime());
    }
}
