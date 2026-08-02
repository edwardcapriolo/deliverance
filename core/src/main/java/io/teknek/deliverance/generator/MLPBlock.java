package io.teknek.deliverance.generator;

import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.math.VectorMath;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.model.tensorparallel.TensorParallelMlp;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensorlib.TensorPlan;

import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;
import java.util.stream.IntStream;
import com.codahale.metrics.Timer;

/**
 * A standard Multi Layer Perceptron block for Transformer models
 */
public class MLPBlock implements FeedForward {
    private final AbstractModel model;
    private final Optional<AbstractTensor> fullyConnectedBias;
    private final AbstractTensor fullyConnectedWeights;

    private final Optional<AbstractTensor> projectionBias;
    private final AbstractTensor projectionWeights;

    private final AbstractTensor upProjectionWeights;

    private final ActivationFunction.Type activationFunction;

    private final AbstractTensor[] batchResults;
    private final AbstractTensor[] batchWeights;
    private final ConfigurableTensorProvider configurableTensorProvider;
    private final String tensorParallelCollectiveKey;

    public MLPBlock(AbstractModel model, ActivationFunction.Type activationFunction, AbstractTensor fullyConnectedWeights,
            AbstractTensor projectionWeights, AbstractTensor upProjectionWeights, ConfigurableTensorProvider configurableTensorProvider) {
        this(model, activationFunction, fullyConnectedWeights, projectionWeights, upProjectionWeights,
                configurableTensorProvider, null);
    }

    public MLPBlock(AbstractModel model, ActivationFunction.Type activationFunction, AbstractTensor fullyConnectedWeights,
            AbstractTensor projectionWeights, AbstractTensor upProjectionWeights,
            ConfigurableTensorProvider configurableTensorProvider, String tensorParallelCollectiveKey) {
        this(model, activationFunction, Optional.empty(), fullyConnectedWeights,
                Optional.empty(), projectionWeights, upProjectionWeights, configurableTensorProvider, tensorParallelCollectiveKey);
    }

    public MLPBlock(
            AbstractModel model,
            ActivationFunction.Type activationFunction,
            AbstractTensor fullyConnectedBias,
            AbstractTensor fullyConnectedWeights,
            AbstractTensor projectionBias,
            AbstractTensor projectionWeights,
            ConfigurableTensorProvider configurableTensorProvider
    ) {

        this(
                model,
                activationFunction,
                Optional.of(fullyConnectedBias),
                fullyConnectedWeights,
                Optional.of(projectionBias),
                projectionWeights,
                null,
                configurableTensorProvider,
                null
        );
    }


    public MLPBlock(
            AbstractModel model,
            ActivationFunction.Type activationFunction,
            Optional<AbstractTensor> fullyConnectedBias,
            AbstractTensor fullyConnectedWeights,
            Optional<AbstractTensor> projectionBias,
            AbstractTensor projectionWeights,
            AbstractTensor upProjectionWeights,
            ConfigurableTensorProvider configurableTensorProvider,
            String tensorParallelCollectiveKey
    ) {
        this.model = model;
        this.activationFunction = activationFunction;
        this.fullyConnectedBias = fullyConnectedBias;
        this.fullyConnectedWeights = fullyConnectedWeights;
        this.projectionBias = projectionBias;
        this.projectionWeights = projectionWeights;
        this.upProjectionWeights = upProjectionWeights;
        this.batchResults = new AbstractTensor[2];
        this.batchWeights = new AbstractTensor[] { fullyConnectedWeights, upProjectionWeights };
        this.configurableTensorProvider = configurableTensorProvider;
        this.tensorParallelCollectiveKey = tensorParallelCollectiveKey;

        configurableTensorProvider.get().registerModelTensor(fullyConnectedWeights);
        if (upProjectionWeights != null) {
            configurableTensorProvider.get().registerModelTensor(upProjectionWeights);
        }
        configurableTensorProvider.get().registerModelTensor(projectionWeights);
    }

    // For FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
    @Override
    public AbstractTensor forward(AbstractTensor lnemb, Optional<Consumer<List<AbstractTensor>>> tensorReducer) {
        return forward(lnemb, tensorReducer, ForwardPhase.DECODE);
    }

    public AbstractTensor forward(AbstractTensor lnemb, Optional<Consumer<List<AbstractTensor>>> tensorReducer,
            ForwardPhase phase) {
        try (Timer.Context ignored = InferenceProfiler.timer(model.getMetricRegistry(), "mlpblock.forward").time()) {
        int hiddenLength = model.getConfig().hiddenLength;
        int batchSize = lnemb.shape().first();
        if (usesLocalTensorParallelShard(hiddenLength)) {
            if (tensorParallelCollectiveKey == null) {
                throw new IllegalStateException("Tensor-parallel MLP requires a collective key");
            }
            AbstractTensor reduced;
            try (Timer.Context ignoredTp = InferenceProfiler.timer(model.getMetricRegistry(), "mlpblock.tensor_parallel_forward").time()) {
                reduced = TensorParallelMlp.forward(lnemb, fullyConnectedWeights, upProjectionWeights,
                    projectionWeights, activationFunction, configurableTensorProvider,
                    model,
                    shape -> model.getTensorAllocator().getDirty(model.getWorkingDType(), shape),
                    model.getTensorParallelCollectives(), tensorParallelCollectiveKey);
            }
            projectionBias.ifPresent(bias -> configurableTensorProvider.get().accumulate(reduced, bias, 0,
                    model.getConfig().embeddingLength));
            return reduced;
        }
        if (phase == ForwardPhase.DECODE && upProjectionWeights != null && fullyConnectedBias.isEmpty()
                && model.getWorkingQType() != null) {
            return tensorPlanDecodeMlp(lnemb, tensorReducer);
        }
        try (
                //TODO ensure its ok to use dirty here do we write the entire tensor
                AbstractTensor buf = model.getTensorAllocator().getDirty(model.getWorkingDType(), TensorShape.of(batchSize, hiddenLength));
                AbstractTensor buf2 = model.getTensorAllocator().getDirty(model.getWorkingDType(), TensorShape.of(batchSize, hiddenLength));
        ) {

            batchResults[0] = buf;
            batchResults[1] = buf2;

            try (Timer.Context ignoredGate = InferenceProfiler.timer(model.getMetricRegistry(), "mlpblock.gate_up_projection").time()) {
                if (InferenceProfiler.isEnabled()) {
                    InferenceProfiler.counter(model.getMetricRegistry(), "mlpblock.gate_input_" + lnemb.dType()).inc();
                    InferenceProfiler.counter(model.getMetricRegistry(), "mlpblock.gate_weight_" + fullyConnectedWeights.dType()).inc();
                    if (upProjectionWeights != null) {
                        InferenceProfiler.counter(model.getMetricRegistry(), "mlpblock.up_weight_" + upProjectionWeights.dType()).inc();
                    }
                }
                io.teknek.deliverance.tensor.operations.TensorOperations projectionOps =
                        projectionOperations(lnemb, fullyConnectedWeights, phase);
                VectorMath.pchunk(0, hiddenLength, (chunkStart, chunkSize) -> {
                if (upProjectionWeights != null) {
                    projectionOps
                            .dotProductBatchChunk(batchResults, lnemb, batchWeights, 0,
                                    model.getConfig().embeddingLength, chunkStart, chunkSize);
                } else {
                    projectionOps
                            .dotProductChunk(buf, lnemb, fullyConnectedWeights, 0, model.getConfig().embeddingLength, chunkStart, chunkSize);
                }
                }, projectionOps.parallelSplitSize(), model.getPool());
            }

            fullyConnectedBias.ifPresent(
                    bias -> configurableTensorProvider.get().accumulate(buf, bias, 0, hiddenLength)
            );

            if (upProjectionWeights != null) {
                model.getMetricRegistry().counter("mlpblock.tensorplan.fused_activation_multiply").inc();
                if (InferenceProfiler.isEnabled()) {
                    InferenceProfiler.counter(model.getMetricRegistry(), "mlpblock.tensorplan.fused_activation_multiply").inc();
                }
                // TensorOperations can fuse activation*multiply with block quantization, avoiding a full F32
                // intermediate pass before the down projection.
            } else {
                // Not using pfor because we can use all cores
                try (Timer.Context ignoredActivation = InferenceProfiler.timer(model.getMetricRegistry(), "mlpblock.activation").time()) {
                    IntStream.range(0, hiddenLength).parallel().forEach(i -> {
                    for (int j = 0; j < batchSize; j++) {
                        float w1 = buf.get(j, i);
                        float w1a = ActivationFunction.eval(activationFunction, w1);
                        buf.set(w1a, j, i);
                    }
                    });
                }
            }

            if (InferenceProfiler.isEnabled()) {
                InferenceProfiler.counter(model.getMetricRegistry(), "mlpblock.down_input_" + buf.dType()).inc();
                InferenceProfiler.counter(model.getMetricRegistry(), "mlpblock.down_weight_" + projectionWeights.dType()).inc();
            }
            try (AbstractTensor bufq = hiddenForDownProjection(buf, buf2, upProjectionWeights != null)) {
                // matmul the projection and sum into input
                AbstractTensor result = model.makeTensor(batchSize, model.getConfig().embeddingLength);
                try (Timer.Context ignoredDown = InferenceProfiler.timer(model.getMetricRegistry(), "mlpblock.down_projection").time()) {
                    io.teknek.deliverance.tensor.operations.TensorOperations downOps =
                            projectionOperations(bufq, projectionWeights, phase);
                    VectorMath.pchunk(0, model.getConfig().embeddingLength, (chunkStart, chunkSize) -> {
                    downOps
                            .dotProductChunk(
                                    result,
                                    bufq,
                                    projectionWeights,
                                    0,
                                    hiddenLength,
                                    chunkStart,
                                    chunkSize
                            );
                    }, downOps.parallelSplitSize(), model.getPool());
                }

                tensorReducer.ifPresent(func -> func.accept(Collections.singletonList(result)));

                projectionBias.ifPresent(bias -> configurableTensorProvider.get().accumulate(result, bias, 0, model.getConfig().embeddingLength));
                return result;
            }
        }
        }
    }

    private AbstractTensor tensorPlanDecodeMlp(AbstractTensor lnemb, Optional<Consumer<List<AbstractTensor>>> tensorReducer) {
        if (InferenceProfiler.isEnabled()) {
            InferenceProfiler.counter(model.getMetricRegistry(), "mlpblock.tensorplan.decode_full_mlp").inc();
            InferenceProfiler.counter(model.getMetricRegistry(), "mlpblock.gate_input_" + lnemb.dType()).inc();
            InferenceProfiler.counter(model.getMetricRegistry(), "mlpblock.gate_weight_" + fullyConnectedWeights.dType()).inc();
            InferenceProfiler.counter(model.getMetricRegistry(), "mlpblock.up_weight_" + upProjectionWeights.dType()).inc();
            InferenceProfiler.counter(model.getMetricRegistry(), "mlpblock.down_input_" + model.getWorkingQType()).inc();
            InferenceProfiler.counter(model.getMetricRegistry(), "mlpblock.down_weight_" + projectionWeights.dType()).inc();
        }
        TensorPlan plan = TensorPlanSupport.plan(model, configurableTensorProvider.get());
        AbstractTensor result = plan.input("input", lnemb)
                .mlp(plan.immutable("gateWeight", fullyConnectedWeights),
                        plan.immutable("upWeight", upProjectionWeights),
                        plan.immutable("downWeight", projectionWeights),
                        activationFunction,
                        model.getWorkingQType())
                .as("mlpOutput")
                .materialize();
        tensorReducer.ifPresent(func -> func.accept(Collections.singletonList(result)));
        projectionBias.ifPresent(bias -> configurableTensorProvider.get().accumulate(result, bias, 0,
                model.getConfig().embeddingLength));
        return result;
    }

    private AbstractTensor downQuantize(AbstractTensor buf) {
        try (Timer.Context ignored = InferenceProfiler.timer(model.getMetricRegistry(), "mlpblock.down_quantize").time()) {
            return model.maybeQuantize(buf);
        }
    }

    private io.teknek.deliverance.tensor.operations.TensorOperations projectionOperations(AbstractTensor input,
            AbstractTensor weight, ForwardPhase phase) {
        io.teknek.deliverance.tensor.operations.TensorOperations operations =
                model.prefillProjectionOperations(input, weight, phase);
        return operations == null ? configurableTensorProvider.get() : operations;
    }

    private AbstractTensor hiddenForDownProjection(AbstractTensor gate, AbstractTensor up, boolean fusedGateUp) {
        if (!fusedGateUp) {
            return downQuantize(gate);
        }
        if (model.getWorkingQType() == null) {
            return activationMultiply(gate, up);
        }
        try (Timer.Context ignored = InferenceProfiler.timer(model.getMetricRegistry(),
                "mlpblock.fused_activation_multiply_quantize").time()) {
            registerTensorRuntimeCounters();
            TensorPlan plan = TensorPlanSupport.plan(model, configurableTensorProvider.get());
            return plan.input("gate", gate)
                    .activate(activationFunction)
                    .multiply(plan.input("up", up))
                    .quantize(model.getWorkingQType())
                    .as("hiddenQ")
                    .materialize();
        }
    }

    private AbstractTensor activationMultiply(AbstractTensor gate, AbstractTensor up) {
        int hiddenLength = (int) gate.shape().last();
        int batchSize = (int) gate.shape().first();
        AbstractTensor hidden = new FloatBufferTensor(batchSize, hiddenLength);
        IntStream.range(0, hiddenLength).parallel().forEach(i -> {
            for (int row = 0; row < batchSize; row++) {
                hidden.set(ActivationFunction.eval(activationFunction, gate.get(row, i)) * up.get(row, i), row, i);
            }
        });
        return hidden;
    }

    private void registerTensorRuntimeCounters() {
        if (!InferenceProfiler.isEnabled()) {
            return;
        }
        InferenceProfiler.counter(model.getMetricRegistry(), "tensorruntime.locality.local").inc(0);
        InferenceProfiler.counter(model.getMetricRegistry(), "tensorruntime.locality.remote").inc(0);
        InferenceProfiler.counter(model.getMetricRegistry(), "tensorruntime.locality.unknown").inc(0);
        InferenceProfiler.counter(model.getMetricRegistry(), "tensorruntime.affinity.pinned").inc(0);
        InferenceProfiler.counter(model.getMetricRegistry(), "tensorruntime.affinity.failed").inc(0);
        InferenceProfiler.counter(model.getMetricRegistry(), "tensorruntime.affinity.unsupported").inc(0);
    }

    private boolean usesLocalTensorParallelShard(int hiddenLength) {
        return model.getTensorParallelContext().enabled()
                && upProjectionWeights != null
                && fullyConnectedWeights.shape().first() != hiddenLength;
    }
}
