package io.teknek.deliverance.generator2;

import io.dropwizard.metrics5.Timer;
import io.teknek.deliverance.generator.ForwardPhase;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.BatchDotProduct;
import io.teknek.deliverance.tensor2.DotProductBatchChunk;
import io.teknek.deliverance.tensor2.ActivationMultiplyQuantize;
import io.teknek.deliverance.tensor2.MultiplyAccumulate;
import io.teknek.deliverance.tensor2.TensorRef;
import io.teknek.deliverance.safetensors.LoraLayerDelta;

import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;

/** TensorRef-native SwiGLU MLP block used by Qwen3. */
public class MLPBlock2 implements FeedForward2 {
    private final AbstractModel model;
    private final TensorRef gateWeights;
    private final TensorRef upWeights;
    private final TensorRef downWeights;
    private final int hiddenLength;
    private final int embeddingLength;
    private final Lighter lighter;
    private final String gateWeightName;
    private final String upWeightName;
    private final String downWeightName;

    public MLPBlock2(AbstractModel model, TensorRef gateWeights, TensorRef upWeights, TensorRef downWeights,
            Lighter lighter) {
        this(model, gateWeights, upWeights, downWeights, lighter, null, null, null);
    }

    public MLPBlock2(AbstractModel model, TensorRef gateWeights, TensorRef upWeights, TensorRef downWeights,
            Lighter lighter, String gateWeightName, String upWeightName, String downWeightName) {
        this.model = java.util.Objects.requireNonNull(model, "model");
        this.gateWeights = java.util.Objects.requireNonNull(gateWeights, "gateWeights");
        this.upWeights = java.util.Objects.requireNonNull(upWeights, "upWeights");
        this.downWeights = java.util.Objects.requireNonNull(downWeights, "downWeights");
        this.hiddenLength = model.getConfig().hiddenLength;
        this.embeddingLength = model.getConfig().embeddingLength;
        this.lighter = java.util.Objects.requireNonNull(lighter, "lighter");
        this.gateWeightName = gateWeightName;
        this.upWeightName = upWeightName;
        this.downWeightName = downWeightName;
    }

    @Override
    public TensorRef forward(TensorRef input, Optional<Consumer<List<TensorRef>>> tensorReducer) {
        return forward(input, tensorReducer, ForwardPhase.DECODE);
    }

    public TensorRef forward(TensorRef input) {
        return forward(input, Optional.empty(), ForwardPhase.DECODE);
    }

    @Override
    public TensorRef forward(TensorRef input, Optional<Consumer<List<TensorRef>>> tensorReducer, ForwardPhase phase) {
        int batchSize = input.shape().first();
        if (model.getTensorParallelContext().enabled() && gateWeights.shape().first() != hiddenLength) {
            throw new UnsupportedOperationException("TensorRef tensor-parallel MLP is not ported");
        }
        TensorRef gate = model.makeDenseTensorRef(batchSize, hiddenLength);
        TensorRef up = model.makeDenseTensorRef(batchSize, hiddenLength);
        TensorRef output = model.makeDenseTensorRef(batchSize, embeddingLength);
        TensorRef downInput = null;
        try (Timer.Context ignored = InferenceProfiler.timer(model.getMetricRegistry(), "mlpblock2.forward").time()) {
            try (Timer.Context ignoredGate = InferenceProfiler.timer(model.getMetricRegistry(),
                    "mlpblock2.gate_up_projection").time()) {
                if (InferenceProfiler.isEnabled()) {
                    InferenceProfiler.counter(model.getMetricRegistry(), "mlpblock2.gate_input_" + input.dType()).inc();
                    InferenceProfiler.counter(model.getMetricRegistry(), "mlpblock2.gate_weight_" + gateWeights.dType()).inc();
                    InferenceProfiler.counter(model.getMetricRegistry(), "mlpblock2.up_weight_" + upWeights.dType()).inc();
                }
                model.runChunks("mlpblock2.gate_up_projection", 0, hiddenLength,
                        model.primaryTensorOperations().parallelSplitSize(), Optional.empty(), (chunkStart, chunkSize) -> {
                    DotProductBatchChunk paired = new DotProductBatchChunk()
                            .results(gate, up)
                            .input(input)
                            .weights(gateWeights, upWeights)
                            .inputColumnStart(0)
                            .weightColumnStart(0)
                            .columnLength(embeddingLength)
                            .weightRowStart(chunkStart)
                            .weightRowCount(chunkSize)
                            .outputColumnStart(chunkStart);
                    if (lighter.supportsDotProductBatchChunk(paired)) {
                        model.getCompositeOps().dotProductBatchChunk(paired);
                    } else {
                        projectChunk(gate, input, gateWeights, embeddingLength, chunkStart, chunkSize, phase);
                        projectChunk(up, input, upWeights, embeddingLength, chunkStart, chunkSize, phase);
                    }
                });
            }
            applyLora(gateWeightName, gate, input, phase, "mlpblock2.gate_lora");
            applyLora(upWeightName, up, input, phase, "mlpblock2.up_lora");
            ActivationMultiplyQuantize fused = new ActivationMultiplyQuantize(gate, up,
                    model.getConfig().activationFunction, model.getWorkingQType())
                    .offsetAndLength(0, hiddenLength);
            if (model.getCompositeOps().supportsActivationMultiplyQuantize(fused)) {
                downInput = model.getCompositeOps().activationMultiplyQuantize(fused);
            } else {
                activate(gate, model.getConfig().activationFunction);
                lighter.multiplyAccumulate(new MultiplyAccumulate(up).into(gate).offsetAndLength(0, hiddenLength));
                downInput = gate.dType() == model.getWorkingQType()
                        ? gate
                        : lighter.reshape(gate, model.getWorkingQType());
            }
            if (InferenceProfiler.isEnabled()) {
                InferenceProfiler.counter(model.getMetricRegistry(), "mlpblock2.down_input_" + downInput.dType()).inc();
                InferenceProfiler.counter(model.getMetricRegistry(), "mlpblock2.down_weight_" + downWeights.dType()).inc();
            }
            TensorRef downInputForChunks = downInput;
            try (Timer.Context ignoredDown = InferenceProfiler.timer(model.getMetricRegistry(),
                    "mlpblock2.down_projection").time()) {
                model.runChunks("mlpblock2.down_projection", 0, embeddingLength,
                        model.primaryTensorOperations().parallelSplitSize(), Optional.empty(), (chunkStart, chunkSize) ->
                                projectChunk(output, downInputForChunks, downWeights, hiddenLength, chunkStart,
                                 chunkSize, phase));
            }
            applyLora(downWeightName, output, downInputForChunks, phase,
                    "mlpblock2.down_lora");
            tensorReducer.ifPresent(func -> func.accept(Collections.singletonList(output)));
            return output;
        } catch (RuntimeException | Error e) {
            output.close();
            throw e;
        } finally {
            if (downInput != null && downInput != gate) {
                downInput.close();
            }
            gate.close();
            up.close();
        }
    }

    private void projectChunk(TensorRef output, TensorRef input, TensorRef weight, int inputLength,
            int chunkStart, int chunkSize, ForwardPhase phase) {
        if (phase == ForwardPhase.PREFILL && input.shape().first() > 1) {
            lighter.batchDotProduct(new BatchDotProduct()
                    .result(output).a(input).b(weight)
                    .aColumnOffset(0).bColumnOffset(0).columnLength(inputLength)
                    .resultRowOffset(0).bRowOffset(chunkStart).rowChunkSize(chunkSize));
        } else {
            lighter.dotProductRows(output, input, weight, 0, inputLength, chunkStart, chunkSize, chunkStart);
        }
    }

    private void activate(TensorRef target, ActivationFunction.Type activationFunction) {
        int batchSize = target.shape().first();
        model.runChunks("mlpblock2.activation", 0, hiddenLength, model.primaryTensorOperations().parallelSplitSize(),
                Optional.empty(), (chunkStart, chunkSize) -> {
            int chunkEnd = chunkStart + chunkSize;
            for (int column = chunkStart; column < chunkEnd; column++) {
                for (int row = 0; row < batchSize; row++) {
                    target.set(ActivationFunction.eval(activationFunction, target.get(row, column)), row, column);
                }
            }
        });
    }

    private boolean hasActiveLoraDelta() {
        return gateWeightName != null && model.activeLoraDeltaFor(gateWeightName).isPresent()
                || upWeightName != null && model.activeLoraDeltaFor(upWeightName).isPresent()
                || downWeightName != null && model.activeLoraDeltaFor(downWeightName).isPresent();
    }

    private void applyLora(String weightName, TensorRef output, TensorRef input,
            ForwardPhase phase, String metricName) {
        if (weightName != null) {
            model.activeLoraDeltaFor(weightName)
                    .ifPresent(value -> LoraDeltaApplier2.apply(model, lighter, output, input, value, phase, metricName));
        }
    }
}
