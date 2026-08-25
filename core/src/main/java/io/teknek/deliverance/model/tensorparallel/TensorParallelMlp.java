package io.teknek.deliverance.model.tensorparallel;

import io.dropwizard.metrics5.Timer;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.math.VectorMath;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;

import java.util.function.Function;
import java.util.Optional;
import java.util.stream.IntStream;

/**
 * Local tensor-parallel MLP math for one rank.
 *
 * <p>The caller supplies dense local shards: gate/up projection shards with shape {@code [localHidden, embedding]} and
 * down projection shard with shape {@code [embedding, localHidden]}. The returned tensor is this rank's partial output;
 * partial outputs from all ranks must be summed to recover the full MLP output.</p>
 */
public final class TensorParallelMlp {
    private TensorParallelMlp() {
    }

    public static AbstractTensor forwardPartial(AbstractTensor input,
            AbstractTensor gateProjectionWeights,
            AbstractTensor upProjectionWeights,
            AbstractTensor downProjectionWeights,
            ActivationFunction.Type activationFunction,
            ConfigurableTensorProvider tensorProvider,
            Function<TensorShape, AbstractTensor> tensorFactory) {
        int batchSize = input.shape().first();
        int embeddingLength = input.shape().last();
        int localHiddenLength = gateProjectionWeights.shape().first();
        validate(input, gateProjectionWeights, upProjectionWeights, downProjectionWeights, embeddingLength, localHiddenLength);

        try (AbstractTensor gate = tensorFactory.apply(TensorShape.of(batchSize, localHiddenLength));
             AbstractTensor up = tensorFactory.apply(TensorShape.of(batchSize, localHiddenLength))) {
            AbstractTensor[] results = new AbstractTensor[]{gate, up};
            AbstractTensor[] weights = new AbstractTensor[]{gateProjectionWeights, upProjectionWeights};
            tensorProvider.get().dotProductBatchChunk(results, input, weights, 0, embeddingLength, 0, localHiddenLength);

            IntStream.range(0, localHiddenLength).parallel().forEach(i -> {
                for (int row = 0; row < batchSize; row++) {
                    gate.set(ActivationFunction.eval(activationFunction, gate.get(row, i)), row, i);
                }
            });
            tensorProvider.get().maccumulate(gate, up, 0, localHiddenLength);

            AbstractTensor partial = tensorFactory.apply(TensorShape.of(batchSize, embeddingLength));
            tensorProvider.get().dotProductChunk(partial, gate, downProjectionWeights, 0, localHiddenLength, 0,
                    embeddingLength);
            return partial;
        }
    }

    public static AbstractTensor forwardPartial(AbstractTensor input,
            AbstractTensor gateProjectionWeights,
            AbstractTensor upProjectionWeights,
            AbstractTensor downProjectionWeights,
            ActivationFunction.Type activationFunction,
            ConfigurableTensorProvider tensorProvider,
            AbstractModel model,
            Function<TensorShape, AbstractTensor> tensorFactory) {
        return forwardPartial(input, gateProjectionWeights, upProjectionWeights, downProjectionWeights,
                activationFunction, tensorProvider, model, tensorFactory, -1);
    }

    public static AbstractTensor forwardPartial(AbstractTensor input,
            AbstractTensor gateProjectionWeights,
            AbstractTensor upProjectionWeights,
            AbstractTensor downProjectionWeights,
            ActivationFunction.Type activationFunction,
            ConfigurableTensorProvider tensorProvider,
            AbstractModel model,
            Function<TensorShape, AbstractTensor> tensorFactory,
            int layerIndex) {
        int batchSize = input.shape().first();
        int embeddingLength = input.shape().last();
        int localHiddenLength = gateProjectionWeights.shape().first();
        validate(input, gateProjectionWeights, upProjectionWeights, downProjectionWeights, embeddingLength, localHiddenLength);

        try (AbstractTensor gate = tensorFactory.apply(TensorShape.of(batchSize, localHiddenLength));
             AbstractTensor up = tensorFactory.apply(TensorShape.of(batchSize, localHiddenLength))) {
            AbstractTensor[] results = new AbstractTensor[]{gate, up};
            AbstractTensor[] weights = new AbstractTensor[]{gateProjectionWeights, upProjectionWeights};
            model.emitLayerDebug(layerIndex, "mlp_projection_input", input);
            try (Timer.Context ignoredGate = InferenceProfiler.timer(model.getMetricRegistry(), "tensorparallelmlp.gate_up_projection").time()) {
                if (model.isInWorkingQuantizedType(input)) {
                    model.runChunks("tensorparallelmlp.gate_up_projection", 0, localHiddenLength,
                            tensorProvider.get().parallelSplitSize(), Optional.of(input), (chunkStart, chunkSize) ->
                            tensorProvider.get().dotProductBatchChunk(results, input, weights, 0, embeddingLength,
                                    chunkStart, chunkSize));
                } else {
                    try (AbstractTensor inputq = inputQuantize(model, input)) {
                        model.runChunks("tensorparallelmlp.gate_up_projection", 0, localHiddenLength,
                                tensorProvider.get().parallelSplitSize(), Optional.of(inputq), (chunkStart, chunkSize) ->
                                tensorProvider.get().dotProductBatchChunk(results, inputq, weights, 0, embeddingLength,
                                        chunkStart, chunkSize));
                    }
                }
            }
            model.emitLayerDebug(layerIndex, "mlp_gate_projection", gate);
            model.emitLayerDebug(layerIndex, "mlp_up_projection", up);

            try (Timer.Context ignoredActivation = InferenceProfiler.timer(model.getMetricRegistry(), "tensorparallelmlp.activation").time()) {
                model.runChunks("tensorparallelmlp.activation", 0, localHiddenLength,
                        tensorProvider.get().parallelSplitSize(), Optional.of(gate), (chunkStart, chunkSize) -> {
                    int chunkEnd = chunkStart + chunkSize;
                    for (int i = chunkStart; i < chunkEnd; i++) {
                        for (int row = 0; row < batchSize; row++) {
                            gate.set(ActivationFunction.eval(activationFunction, gate.get(row, i)), row, i);
                        }
                    }
                });
            }
            model.emitLayerDebug(layerIndex, "mlp_after_activation", gate);
            try (Timer.Context ignoredMultiply = InferenceProfiler.timer(model.getMetricRegistry(), "tensorparallelmlp.multiply").time()) {
                tensorProvider.get().maccumulate(gate, up, 0, localHiddenLength);
            }
            model.emitLayerDebug(layerIndex, "mlp_after_multiply", gate);

            AbstractTensor partial = tensorFactory.apply(TensorShape.of(batchSize, embeddingLength));
            try (Timer.Context ignoredDown = InferenceProfiler.timer(model.getMetricRegistry(), "tensorparallelmlp.down_projection").time()) {
                if (model.isInWorkingQuantizedType(gate)) {
                    if (InferenceProfiler.isEnabled()) {
                        InferenceProfiler.counter(model.getMetricRegistry(), "tensorparallelmlp.down_input_" + gate.dType()).inc();
                    }
                    model.runChunks("tensorparallelmlp.down_projection", 0, embeddingLength,
                            tensorProvider.get().parallelSplitSize(), Optional.of(gate), (chunkStart, chunkSize) ->
                            tensorProvider.get().dotProductChunk(partial, gate, downProjectionWeights, 0,
                                    localHiddenLength, chunkStart, chunkSize));
                } else {
                    if (InferenceProfiler.isEnabled()) {
                        InferenceProfiler.counter(model.getMetricRegistry(), "tensorparallelmlp.down_input_" + gate.dType()).inc();
                    }
                    try (AbstractTensor gateq = downQuantize(model, gate)) {
                        model.runChunks("tensorparallelmlp.down_projection", 0, embeddingLength,
                                tensorProvider.get().parallelSplitSize(), Optional.of(gateq), (chunkStart, chunkSize) ->
                                tensorProvider.get().dotProductChunk(partial, gateq, downProjectionWeights, 0,
                                        localHiddenLength, chunkStart, chunkSize));
                    }
                }
            }
            model.emitLayerDebug(layerIndex, "mlp_down_partial", partial);
            return partial;
        }
    }

    private static AbstractTensor downQuantize(AbstractModel model, AbstractTensor gate) {
        try (Timer.Context ignored = InferenceProfiler.timer(model.getMetricRegistry(), "tensorparallelmlp.down_quantize").time()) {
            return model.quantizeToWorkingQuantizedType(gate);
        }
    }

    private static AbstractTensor inputQuantize(AbstractModel model, AbstractTensor input) {
        try (Timer.Context ignored = InferenceProfiler.timer(model.getMetricRegistry(), "tensorparallelmlp.input_quantize").time()) {
            return model.quantizeToWorkingQuantizedType(input);
        }
    }

    public static AbstractTensor forward(AbstractTensor input,
            AbstractTensor gateProjectionWeights,
            AbstractTensor upProjectionWeights,
            AbstractTensor downProjectionWeights,
            ActivationFunction.Type activationFunction,
            ConfigurableTensorProvider tensorProvider,
            Function<TensorShape, AbstractTensor> tensorFactory,
            TensorParallelCollectives collectives,
            String collectiveKey) {
        AbstractTensor partial = forwardPartial(input, gateProjectionWeights, upProjectionWeights, downProjectionWeights,
                activationFunction, tensorProvider, tensorFactory);
        AbstractTensor reduced = collectives.allReduceSum(collectiveKey, partial);
        if (reduced != partial) {
            partial.close();
        }
        return reduced;
    }

    public static AbstractTensor forward(AbstractTensor input,
            AbstractTensor gateProjectionWeights,
            AbstractTensor upProjectionWeights,
            AbstractTensor downProjectionWeights,
            ActivationFunction.Type activationFunction,
            ConfigurableTensorProvider tensorProvider,
            AbstractModel model,
            Function<TensorShape, AbstractTensor> tensorFactory,
            TensorParallelCollectives collectives,
            String collectiveKey) {
        return forward(input, gateProjectionWeights, upProjectionWeights, downProjectionWeights, activationFunction,
                tensorProvider, model, tensorFactory, collectives, collectiveKey, -1);
    }

    public static AbstractTensor forward(AbstractTensor input,
            AbstractTensor gateProjectionWeights,
            AbstractTensor upProjectionWeights,
            AbstractTensor downProjectionWeights,
            ActivationFunction.Type activationFunction,
            ConfigurableTensorProvider tensorProvider,
            AbstractModel model,
            Function<TensorShape, AbstractTensor> tensorFactory,
            TensorParallelCollectives collectives,
            String collectiveKey,
            int layerIndex) {
        AbstractTensor partial = forwardPartial(input, gateProjectionWeights, upProjectionWeights, downProjectionWeights,
                activationFunction, tensorProvider, model, tensorFactory, layerIndex);
        AbstractTensor reduced;
        try (Timer.Context ignored = InferenceProfiler.timer(model.getMetricRegistry(), "tensorparallelmlp.all_reduce").time()) {
            reduced = collectives.allReduceSum(collectiveKey, partial);
        }
        model.emitLayerDebug(layerIndex, "mlp_down_reduced", reduced);
        if (reduced != partial) {
            partial.close();
        }
        return reduced;
    }

    private static void validate(AbstractTensor input, AbstractTensor gateProjectionWeights,
            AbstractTensor upProjectionWeights, AbstractTensor downProjectionWeights, int embeddingLength,
            int localHiddenLength) {
        if (gateProjectionWeights.shape().last() != embeddingLength) {
            throw new IllegalArgumentException("gateProjectionWeights must have shape [localHidden, embedding]");
        }
        if (upProjectionWeights.shape().first() != localHiddenLength || upProjectionWeights.shape().last() != embeddingLength) {
            throw new IllegalArgumentException("upProjectionWeights must have shape [localHidden, embedding]");
        }
        if (downProjectionWeights.shape().first() != embeddingLength || downProjectionWeights.shape().last() != localHiddenLength) {
            throw new IllegalArgumentException("downProjectionWeights must have shape [embedding, localHidden]");
        }
    }
}
