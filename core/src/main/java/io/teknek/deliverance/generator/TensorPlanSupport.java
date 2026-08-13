package io.teknek.deliverance.generator;

import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.tensor.operations.TensorOperations;
import io.teknek.deliverance.tensorlib.TensorPlan;
import io.teknek.deliverance.tensorlib.TensorRuntime;
import io.teknek.deliverance.tensorlib.TensorRuntimeGlobal;

final class TensorPlanSupport {
    private TensorPlanSupport() {
    }

    static TensorPlan plan(AbstractModel model, TensorOperations operations) {
        TensorRuntime runtime = TensorRuntimeGlobal.get(model.getMetricRegistry(), model.getTensorRuntimeMode(),
                model.getPool().getCoreCount());
        return new TensorPlan(operations, model.getPool(), model.getMetricRegistry(), model, runtime);
    }
}
