package io.teknek.deliverance.generator;

import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.math.VectorMath;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensorlib.TensorPlan;

import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;
import java.util.stream.IntStream;

public class VariableMLPBlock implements FeedForward {
    private final AbstractModel model;
    private final ActivationFunction.Type activationFunction;
    private final AbstractTensor fullyConnectedWeights;
    private final AbstractTensor projectionWeights;
    private final AbstractTensor upProjectionWeights;
    private final int hiddenLength;
    private final float activationSparsity;
    private final float activationSparsityStdMultiplier;
    private final AbstractTensor[] batchResults = new AbstractTensor[2];
    private final AbstractTensor[] batchWeights = new AbstractTensor[2];
    private final ConfigurableTensorProvider configurableTensorProvider;

    public VariableMLPBlock(
            AbstractModel model,
            ActivationFunction.Type activationFunction,
            AbstractTensor fullyConnectedWeights,
            AbstractTensor projectionWeights,
            AbstractTensor upProjectionWeights,
            int hiddenLength,
            ConfigurableTensorProvider configurableTensorProvider
    ) {
        this(model, activationFunction, fullyConnectedWeights, projectionWeights, upProjectionWeights, hiddenLength,
                0.0f, configurableTensorProvider);
    }

    public VariableMLPBlock(
            AbstractModel model,
            ActivationFunction.Type activationFunction,
            AbstractTensor fullyConnectedWeights,
            AbstractTensor projectionWeights,
            AbstractTensor upProjectionWeights,
            int hiddenLength,
            float activationSparsity,
            ConfigurableTensorProvider configurableTensorProvider
    ) {
        this.model = model;
        this.activationFunction = activationFunction;
        this.fullyConnectedWeights = fullyConnectedWeights;
        this.projectionWeights = projectionWeights;
        this.upProjectionWeights = upProjectionWeights;
        this.hiddenLength = hiddenLength;
        this.activationSparsity = activationSparsity;
        this.activationSparsityStdMultiplier = activationSparsity > 0.0f ? inverseStandardNormal(activationSparsity) : 0.0f;
        this.configurableTensorProvider = configurableTensorProvider;
        this.batchWeights[0] = fullyConnectedWeights;
        this.batchWeights[1] = upProjectionWeights;

        configurableTensorProvider.get().registerModelTensor(fullyConnectedWeights);
        configurableTensorProvider.get().registerModelTensor(upProjectionWeights);
        configurableTensorProvider.get().registerModelTensor(projectionWeights);
    }

    @Override
    public AbstractTensor forward(AbstractTensor input, Optional<Consumer<List<AbstractTensor>>> tensorReducer) {
        return forward(input, tensorReducer, ForwardPhase.DECODE);
    }

    public AbstractTensor forward(AbstractTensor input, Optional<Consumer<List<AbstractTensor>>> tensorReducer,
            ForwardPhase phase) {
        int batchSize = input.shape().first();
        try (
                AbstractTensor gate = model.getTensorAllocator().getDirty(model.getWorkingDType(), TensorShape.of(batchSize, hiddenLength));
                AbstractTensor up = model.getTensorAllocator().getDirty(model.getWorkingDType(), TensorShape.of(batchSize, hiddenLength))
        ) {
            batchResults[0] = gate;
            batchResults[1] = up;
            VectorMath.pchunk(0, hiddenLength, (chunkStart, chunkSize) -> configurableTensorProvider.get()
                    .dotProductBatchChunk(batchResults, input, batchWeights, 0, model.getConfig().embeddingLength, chunkStart, chunkSize),
                    configurableTensorProvider.get().parallelSplitSize(), model.getPool());

            TensorPlan plan = new TensorPlan(configurableTensorProvider.get(), model.getPool(), model.getMetricRegistry());
            applyActivationSparsity(plan, gate, batchSize);
            model.getMetricRegistry().counter("variablemlpblock.tensorplan.intstream_activation_multiply").inc();
            plan.fuseColumnsIntStream("gate", gate.shape())
                    .write("gate", plan.mutable("gate", gate))
                    .read("up", plan.input("up", up))
                    .map("gate = " + activationFunction.name().toLowerCase() + "(gate) * up",
                            TensorPlan.TensorOp.ACTIVATION_MUL_IN_PLACE,
                            (ctx, offset, length) -> applyActivationMultiplyColumns(ctx.tensor("gate"),
                                    ctx.tensor("up"), batchSize, (int) offset))
                    .tensor()
                    .timer("variablemlpblock.multiply")
                    .materialize();

            try (AbstractTensor gateQ = model.maybeQuantize(gate)) {
                AbstractTensor result = model.makeTensor(batchSize, model.getConfig().embeddingLength);
                VectorMath.pchunk(0, model.getConfig().embeddingLength, (chunkStart, chunkSize) -> configurableTensorProvider.get()
                        .dotProductChunk(result, gateQ, projectionWeights, 0, hiddenLength, chunkStart, chunkSize),
                        configurableTensorProvider.get().parallelSplitSize(), model.getPool());
                tensorReducer.ifPresent(func -> func.accept(Collections.singletonList(result)));
                return result;
            }
        }
    }

    private void applyActivationSparsity(TensorPlan plan, AbstractTensor gate, int batchSize) {
        if (activationSparsity <= 0.0f) {
            return;
        }
        model.getMetricRegistry().counter("variablemlpblock.tensorplan.intstream_activation_sparsity").inc();
        plan.fuseRowsIntStream("gate", gate.shape())
                .write("gate", plan.mutable("gate", gate))
                .map("gate = activation_sparsity(gate)", TensorPlan.TensorOp.ACTIVATION_SPARSITY_IN_PLACE,
                        (ctx, offset, length) -> applyActivationSparsityRow(ctx.tensor("gate"), (int) offset))
                .tensor()
                .timer("variablemlpblock.activation_sparsity")
                .materialize();
    }

    private void applyActivationMultiplyColumns(AbstractTensor gate, AbstractTensor up, int batchSize, int column) {
        for (int row = 0; row < batchSize; row++) {
            float activated = ActivationFunction.eval(activationFunction, gate.get(row, column));
            gate.set(activated * up.get(row, column), row, column);
        }
    }

    private void applyActivationSparsityRow(AbstractTensor gate, int row) {
        double sum = 0.0d;
        for (int col = 0; col < hiddenLength; col++) {
            sum += gate.get(row, col);
        }
        double mean = sum / hiddenLength;
        double variance = 0.0d;
        for (int col = 0; col < hiddenLength; col++) {
            double diff = gate.get(row, col) - mean;
            variance += diff * diff;
        }
        double std = Math.sqrt(variance / hiddenLength);
        float cutoff = (float) (mean + std * activationSparsityStdMultiplier);
        for (int col = 0; col < hiddenLength; col++) {
            float v = gate.get(row, col) - cutoff;
            gate.set(Math.max(0.0f, v), row, col);
        }
    }

    static void applyActivationSparsity(AbstractTensor gate, int batchSize, int hiddenLength,
            float activationSparsityStdMultiplier) {
        IntStream.range(0, batchSize).parallel().forEach(row -> {
            double sum = 0.0d;
            for (int col = 0; col < hiddenLength; col++) {
                sum += gate.get(row, col);
            }
            double mean = sum / hiddenLength;
            double variance = 0.0d;
            for (int col = 0; col < hiddenLength; col++) {
                double diff = gate.get(row, col) - mean;
                variance += diff * diff;
            }
            double std = Math.sqrt(variance / hiddenLength);
            float cutoff = (float) (mean + std * activationSparsityStdMultiplier);
            for (int col = 0; col < hiddenLength; col++) {
                float v = gate.get(row, col) - cutoff;
                gate.set(Math.max(0.0f, v), row, col);
            }
        });
    }

    /** Acklam's inverse normal CDF approximation, sufficient for fixed sparsity thresholds. */
    private static float inverseStandardNormal(float p) {
        if (p <= 0.0f || p >= 1.0f) {
            throw new IllegalArgumentException("activation sparsity must be in (0,1)");
        }
        double[] a = {-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
                1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00};
        double[] b = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
                6.680131188771972e+01, -1.328068155288572e+01};
        double[] c = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
                -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00};
        double[] d = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
                3.754408661907416e+00};
        double plow = 0.02425;
        double phigh = 1.0 - plow;
        double q;
        double x;
        if (p < plow) {
            q = Math.sqrt(-2.0 * Math.log(p));
            x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                    / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
        } else if (p <= phigh) {
            q = p - 0.5;
            double r = q * q;
            x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
                    / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
        } else {
            q = Math.sqrt(-2.0 * Math.log(1.0 - p));
            x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                    / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
        }
        return (float) x;
    }
}
