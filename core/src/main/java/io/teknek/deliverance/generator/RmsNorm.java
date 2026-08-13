package io.teknek.deliverance.generator;

import com.codahale.metrics.Histogram;
import com.codahale.metrics.MetricRegistry;
import com.codahale.metrics.Timer;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensorlib.PlannedTensor;
import io.teknek.deliverance.tensorlib.TensorPlan;
import net.jafama.FastMath;

import java.time.Duration;

public class RmsNorm extends LayerNorm {
    private final float weightAdjustment;
    protected Timer totalTime;

    public RmsNorm(AbstractModel m, AbstractTensor weights, MetricRegistry metricRegistry) {
        this(m, weights, 0.0f, metricRegistry);
        totalTime = metricReigstry.timer("rms_norm");
    }

    public RmsNorm(AbstractModel m, AbstractTensor weights, float weightAdjustment, MetricRegistry metricRegistry) {
        super(m, null, weights, metricRegistry);
        totalTime = metricReigstry.timer("rms_norm");
        this.weightAdjustment = weightAdjustment;
    }

    @Override
    public AbstractTensor forward(AbstractTensor input, int offset, int length) {
        try (Timer.Context ignored = InferenceProfiler.timer(metricReigstry, "rmsnorm.forward").time()) {
        long start = System.currentTimeMillis();
        AbstractTensor output = model.makeDenseTensor(input.shape());
        int limit = offset + length;
        if (model.getConfigurableTensorProvider() == null) {
            applyRmsNorm(input, output, offset, length, limit);
            long end = System.currentTimeMillis();
            totalTime.update(Duration.ofMillis(end-start));
            return output;
        }
        // RMSNorm is a tiny row-local operation during decode; TensorRuntime scheduling and locality checks cost more
        // than they save here, so keep TensorPlan diagnostics but execute inline.
        TensorPlan plan = TensorPlanSupport.plan(model, model.getConfigurableTensorProvider().get())
                .forcedRunMode(TensorPlan.RunMode.CALLER_THREAD);
        TensorPlan.Tensor planned = plan.fuseRowsIntStream("rmsnorm", output.shape())
                .read("input", plan.input("input", input))
                .write("output", plan.mutable("output", output))
                .map("output = rmsnorm(input)", TensorPlan.TensorOp.CUSTOM, (ctx, rowOffset, rowLength) -> {
                    AbstractTensor in = ctx.tensor("input");
                    AbstractTensor out = ctx.tensor("output");
                    applyRmsNormRow(in, out, (int) rowOffset, offset, length, limit);
                })
                .tensor();
        model.traceTensorPlan(plan.ownerClass(), "rmsnorm.forward", "UNKNOWN", -1, plan.runMode().name(),
                planned.plan());
        planned.materialize();
        long end = System.currentTimeMillis();
        totalTime.update(Duration.ofMillis(end-start));
        return output;
        }
    }

    @Override
    public PlannedTensor forward(PlannedTensor input) {
        try (Timer.Context ignored = InferenceProfiler.timer(metricReigstry, "rmsnorm.forward").time()) {
        long start = System.currentTimeMillis();
        AbstractTensor inputTensor = input.tensor();
        AbstractTensor output = model.makeDenseTensor(inputTensor.shape());
        int offset = 0;
        int length = model.getConfig().embeddingLength;
        int limit = offset + length;
        if (model.getConfigurableTensorProvider() == null) {
            applyRmsNorm(inputTensor, output, offset, length, limit);
            long end = System.currentTimeMillis();
            totalTime.update(Duration.ofMillis(end-start));
            return new PlannedTensor(output, input.plan());
        }
        TensorPlan plan = TensorPlanSupport.plan(model, model.getConfigurableTensorProvider().get())
                .forcedRunMode(TensorPlan.RunMode.CALLER_THREAD);
        TensorPlan.Tensor planned = plan.fuseRowsIntStream("rmsnorm", output.shape())
                .read("input", plan.input("input", input.plan(), inputTensor))
                .write("output", plan.mutable("output", output))
                .map("output = rmsnorm(input)", TensorPlan.TensorOp.CUSTOM, (ctx, rowOffset, rowLength) -> {
                    AbstractTensor in = ctx.tensor("input");
                    AbstractTensor out = ctx.tensor("output");
                    applyRmsNormRow(in, out, (int) rowOffset, offset, length, limit);
                })
                .tensor()
                .as("rmsnorm.output");
        model.traceTensorPlan(plan.ownerClass(), "rmsnorm.forward", "UNKNOWN", -1, plan.runMode().name(),
                planned.plan());
        planned.materialize();
        long end = System.currentTimeMillis();
        totalTime.update(Duration.ofMillis(end-start));
        return new PlannedTensor(output, planned);
        }
    }

    private void applyRmsNorm(AbstractTensor input, AbstractTensor output, int offset, int length, int limit) {
        int batchSize = input.shape().first();
        for (int b = 0; b < batchSize; b++) {
            applyRmsNormRow(input, output, b, offset, length, limit);
        }
    }

    private void applyRmsNormRow(AbstractTensor input, AbstractTensor output, int b, int offset, int length, int limit) {
        double ss = 0.0f;
        for (int j = offset; j < limit; j++) {
            float v = input.get(b, j);
            ss += v * v;
        }
        //originally normalizaing over the enter length
        //ss /= model.getConfig().embeddingLength;
        ss /= length;
        ss += model.getConfig().layerNormEps;
        ss = (1.0 / FastMath.sqrt(ss));
        for (int j = offset; j < limit; j++) {
            output.set((weightAdjustment + weights.get(0, j)) * ((float) ss * input.get(b, j)), b, j);
        }
    }
}
