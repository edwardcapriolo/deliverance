package io.teknek.deliverance.generator;

import com.codahale.metrics.Histogram;
import com.google.common.base.Preconditions;
import io.teknek.deliverance.CausualWhisperer;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensorlib.TensorPlan;
import net.jafama.FastMath;
import com.codahale.metrics.MetricRegistry;
/*
* https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
* */
public class LayerNorm {

    protected final AbstractModel model;
    private final AbstractTensor bias;
    protected final AbstractTensor weights;
    protected final MetricRegistry metricReigstry;
    public final Histogram totalTime;

    public LayerNorm(AbstractModel m, AbstractTensor bias, AbstractTensor weights, MetricRegistry parent) {
        this.model = m;
        this.bias = bias;
        this.weights = weights;
        this.metricReigstry = parent;
        totalTime = metricReigstry.histogram("layer_norm");
    }

    public AbstractTensor forward(AbstractTensor input) {
        Preconditions.checkArgument(input.shape().dims() == 2);
        int size = input.shape().last();
        Preconditions.checkArgument(size == model.getConfig().embeddingLength);
        return forward(input, 0, model.getConfig().embeddingLength);
    }

    public AbstractTensor forward(AbstractTensor input, int offset, int length) {
        long start = System.currentTimeMillis();
        AbstractTensor output = model.getTensorAllocator().getDirty(input.dType(), input.shape());
        if (model.getConfigurableTensorProvider() == null) {
            performLayerNorm(input, output, weights, bias, model.getConfig().layerNormEps, offset, length,
                    model.getConfig().embeddingLength);
            long end = System.currentTimeMillis();
            totalTime.update(end - start);
            return output;
        }
        // LayerNorm is a tiny row-local operation during decode; TensorRuntime scheduling and locality checks cost more
        // than they save here, so keep TensorPlan diagnostics but execute inline.
        TensorPlan plan = TensorPlanSupport.plan(model, model.getConfigurableTensorProvider().get())
                .forcedRunMode(TensorPlan.RunMode.CALLER_THREAD);
        int embeddingLength = model.getConfig().embeddingLength;
        int limit = offset + length;
        plan.fuseRowsIntStream("layernorm", output.shape())
                .read("input", plan.input("input", input))
                .write("output", plan.mutable("output", output))
                .map("output = layernorm(input)", TensorPlan.TensorOp.CUSTOM, (ctx, rowOffset, rowLength) -> {
                    AbstractTensor in = ctx.tensor("input");
                    AbstractTensor out = ctx.tensor("output");
                    performLayerNormRow(in, out, weights, bias, model.getConfig().layerNormEps, offset, limit,
                            embeddingLength, (int) rowOffset);
                })
                .tensor()
                .materialize();
        long end = System.currentTimeMillis();
        totalTime.update(end - start);
        return output;
    }

    public static void performLayerNorm(AbstractTensor input, AbstractTensor output, AbstractTensor weights,
                                        AbstractTensor bias, float eps, int offset, int length, int embeddingLength){
        int batchSize = input.shape().first();
        int limit = offset + length;
        for (int row = 0; row < batchSize; row++) {
            performLayerNormRow(input, output, weights, bias, eps, offset, limit, embeddingLength, row);
        }
    }

    static void performLayerNormRow(AbstractTensor input, AbstractTensor output, AbstractTensor weights,
            AbstractTensor bias, float eps, int offset, int limit, int embeddingLength, int row) {
        float sum = 0;
        float sumSq = 0;
        if (row == 3) {
            CausualWhisperer.LOGGER.debug("LayerNorm.forward batch {} loop offset {} to limit {}", row, offset, limit);
        }
        for (int i = offset; i < limit; i++) {
            float v = input.get(row, i);
            sum += v;
            sumSq += v * v;
        }
        float mean = sum / embeddingLength;
        float variance = sumSq / embeddingLength - mean * mean;
        float invStddev = 1.0f / (float) FastMath.sqrt(variance + eps);
        if (row == 3) {
            CausualWhisperer.LOGGER.debug("LayerNorm.forward sum {} sumSq {} mean {} variance {} invStdDev {} ",
                    sum, sumSq, mean, variance, invStddev);
        }
        for (int i = offset; i < limit; i++) {
            float v = (input.get(row, i) - mean) * invStddev * weights.get(0, i) + bias.get(0, i);
            output.set(v, row, i);
        }
    }
}
