package io.teknek.deliverance.model.tensorparallel;

import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.math.VectorMath;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;

import java.util.function.Function;
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
            AbstractModel model,
            Function<TensorShape, AbstractTensor> tensorFactory) {
        int batchSize = input.shape().first();
        int embeddingLength = input.shape().last();
        int localHiddenLength = gateProjectionWeights.shape().first();
        validate(input, gateProjectionWeights, upProjectionWeights, downProjectionWeights, embeddingLength, localHiddenLength);

        try (AbstractTensor gate = tensorFactory.apply(TensorShape.of(batchSize, localHiddenLength));
             AbstractTensor up = tensorFactory.apply(TensorShape.of(batchSize, localHiddenLength))) {
            AbstractTensor[] results = new AbstractTensor[]{gate, up};
            AbstractTensor[] weights = new AbstractTensor[]{gateProjectionWeights, upProjectionWeights};
            InferenceProfiler.time("tp_mlp.gate_up_projection", () -> {
                if (model.isInWorkingQuantizedType(input)) {
                    VectorMath.pchunk(0, localHiddenLength, (chunkStart, chunkSize) ->
                            tensorProvider.get().dotProductBatchChunk(results, input, weights, 0, embeddingLength,
                                    chunkStart, chunkSize), tensorProvider.get().parallelSplitSize(), model.getPool());
                } else {
                    try (AbstractTensor inputq = InferenceProfiler.time("tp_mlp.input_quantize",
                            () -> model.quantizeToWorkingQuantizedType(input))) {
                        VectorMath.pchunk(0, localHiddenLength, (chunkStart, chunkSize) ->
                                tensorProvider.get().dotProductBatchChunk(results, inputq, weights, 0, embeddingLength,
                                        chunkStart, chunkSize), tensorProvider.get().parallelSplitSize(), model.getPool());
                    }
                }
                return null;
            });

            InferenceProfiler.time("tp_mlp.activation", () -> {
                IntStream.range(0, localHiddenLength).parallel().forEach(i -> {
                for (int row = 0; row < batchSize; row++) {
                    gate.set(ActivationFunction.eval(activationFunction, gate.get(row, i)), row, i);
                }
                });
                return null;
            });
            InferenceProfiler.time("tp_mlp.multiply", () -> {
                tensorProvider.get().maccumulate(gate, up, 0, localHiddenLength);
                return null;
            });

            AbstractTensor partial = tensorFactory.apply(TensorShape.of(batchSize, embeddingLength));
            InferenceProfiler.time("tp_mlp.down_projection", () -> {
                if (model.isInWorkingQuantizedType(gate)) {
                    if (InferenceProfiler.isEnabled()) {
                        model.getMetricRegistry().counter("tp_mlp.down_quantize.input_dtype." + gate.dType()).inc();
                    }
                    VectorMath.pchunk(0, embeddingLength, (chunkStart, chunkSize) ->
                            tensorProvider.get().dotProductChunk(partial, gate, downProjectionWeights, 0,
                                    localHiddenLength, chunkStart, chunkSize), tensorProvider.get().parallelSplitSize(),
                            model.getPool());
                } else {
                    if (InferenceProfiler.isEnabled()) {
                        model.getMetricRegistry().counter("tp_mlp.down_quantize.input_dtype." + gate.dType()).inc();
                    }
                    try (AbstractTensor gateq = InferenceProfiler.time("tp_mlp.down_quantize",
                            () -> model.quantizeToWorkingQuantizedType(gate))) {
                        VectorMath.pchunk(0, embeddingLength, (chunkStart, chunkSize) ->
                                tensorProvider.get().dotProductChunk(partial, gateq, downProjectionWeights, 0,
                                        localHiddenLength, chunkStart, chunkSize), tensorProvider.get().parallelSplitSize(),
                                model.getPool());
                    }
                }
                return null;
            });
            return partial;
        }
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
        AbstractTensor partial = forwardPartial(input, gateProjectionWeights, upProjectionWeights, downProjectionWeights,
                activationFunction, tensorProvider, model, tensorFactory);
        AbstractTensor reduced = InferenceProfiler.time("tp_mlp.all_reduce", () -> collectives.allReduceSum(collectiveKey, partial));
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
