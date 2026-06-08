package io.teknek.deliverance.model.tensorparallel;

import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorShape;
import net.jafama.FastMath;

import java.util.function.Function;

/**
 * Correctness-first local tensor-parallel causal attention math for one rank.
 *
 * <p>This helper intentionally models the core tensor-parallel identity without RoPE, KV cache paging, or GQA. It is a
 * staging point for proving that split attention heads plus a reduced output projection can match a full attention
 * computation before the production attention path is changed.</p>
 */
public final class TensorParallelAttention {
    private TensorParallelAttention() {
    }

    /**
     * Computes this rank's local attention-output partial.
     *
     * <p>Inputs and weights are local dense tensors. The Q/K/V projection weights must have shape
     * {@code [localHeads * headDim, embedding]}. The output projection weight must have shape
     * {@code [embedding, localHeads * headDim]}. The returned tensor has shape {@code [sequence, embedding]} and must be
     * summed with every other rank's partial output to reconstruct the full attention output.</p>
     *
     * <p>Current intentional limitations: no RoPE, no GQA, no sliding window, no attention softcap, no dropout, and no KV
     * cache paging. This method is suitable for proving the tensor-parallel split/reduce identity, not for full model
     * inference.</p>
     */
    public static AbstractTensor forwardPartial(AbstractTensor input,
            AbstractTensor queryProjectionWeights,
            AbstractTensor keyProjectionWeights,
            AbstractTensor valueProjectionWeights,
            AbstractTensor outputProjectionWeights,
            int localHeads,
            int headDim,
            float attentionScale,
            Function<TensorShape, AbstractTensor> tensorFactory) {
        int sequenceLength = input.shape().first();
        int embeddingLength = input.shape().last();
        int localAttentionLength = localHeads * headDim;
        validate(input, queryProjectionWeights, keyProjectionWeights, valueProjectionWeights, outputProjectionWeights,
                localAttentionLength, embeddingLength);

        try (AbstractTensor queries = tensorFactory.apply(TensorShape.of(sequenceLength, localAttentionLength));
             AbstractTensor keys = tensorFactory.apply(TensorShape.of(sequenceLength, localAttentionLength));
             AbstractTensor values = tensorFactory.apply(TensorShape.of(sequenceLength, localAttentionLength));
             AbstractTensor localAttentionOutput = tensorFactory.apply(TensorShape.of(sequenceLength, localAttentionLength))) {
            project(input, queryProjectionWeights, queries);
            project(input, keyProjectionWeights, keys);
            project(input, valueProjectionWeights, values);
            causalAttention(queries, keys, values, localAttentionOutput, localHeads, headDim, attentionScale);
            AbstractTensor partial = tensorFactory.apply(TensorShape.of(sequenceLength, embeddingLength));
            project(localAttentionOutput, outputProjectionWeights, partial);
            return partial;
        }
    }

    /**
     * Computes this rank's local partial and reduces it through {@link TensorParallelCollectives#allReduceSum}.
     *
     * <p>The {@code collectiveKey} must be stable and identical across ranks for the same logical attention output
     * projection, for example {@code layer.3.self_attn.o_proj}. The returned tensor is the reduced attention output and
     * is owned by the caller.</p>
     */
    public static AbstractTensor forward(AbstractTensor input,
            AbstractTensor queryProjectionWeights,
            AbstractTensor keyProjectionWeights,
            AbstractTensor valueProjectionWeights,
            AbstractTensor outputProjectionWeights,
            int localHeads,
            int headDim,
            float attentionScale,
            Function<TensorShape, AbstractTensor> tensorFactory,
            TensorParallelCollectives collectives,
            String collectiveKey) {
        AbstractTensor partial = forwardPartial(input, queryProjectionWeights, keyProjectionWeights,
                valueProjectionWeights, outputProjectionWeights, localHeads, headDim, attentionScale, tensorFactory);
        AbstractTensor reduced = collectives.allReduceSum(collectiveKey, partial);
        if (reduced != partial) {
            partial.close();
        }
        return reduced;
    }

    private static void project(AbstractTensor input, AbstractTensor weights, AbstractTensor output) {
        int rows = input.shape().first();
        int inputWidth = input.shape().last();
        int outputWidth = weights.shape().first();
        for (int row = 0; row < rows; row++) {
            for (int out = 0; out < outputWidth; out++) {
                float sum = 0.0f;
                for (int in = 0; in < inputWidth; in++) {
                    sum += input.get(row, in) * weights.get(out, in);
                }
                output.set(sum, row, out);
            }
        }
    }

    private static void causalAttention(AbstractTensor queries, AbstractTensor keys, AbstractTensor values,
            AbstractTensor output, int localHeads, int headDim, float attentionScale) {
        int sequenceLength = queries.shape().first();
        float[] scores = new float[sequenceLength];
        for (int position = 0; position < sequenceLength; position++) {
            for (int head = 0; head < localHeads; head++) {
                int offset = head * headDim;
                for (int prior = 0; prior <= position; prior++) {
                    float score = 0.0f;
                    for (int dim = 0; dim < headDim; dim++) {
                        score += queries.get(position, offset + dim) * keys.get(prior, offset + dim);
                    }
                    scores[prior] = score * attentionScale;
                }
                softmaxPrefix(scores, position + 1);
                for (int dim = 0; dim < headDim; dim++) {
                    float value = 0.0f;
                    for (int prior = 0; prior <= position; prior++) {
                        value += scores[prior] * values.get(prior, offset + dim);
                    }
                    output.set(value, position, offset + dim);
                }
            }
        }
    }

    private static void softmaxPrefix(float[] values, int length) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < length; i++) {
            max = Math.max(max, values[i]);
        }
        float sum = 0.0f;
        for (int i = 0; i < length; i++) {
            values[i] = (float) FastMath.exp(values[i] - max);
            sum += values[i];
        }
        for (int i = 0; i < length; i++) {
            values[i] /= sum;
        }
    }

    private static void validate(AbstractTensor input,
            AbstractTensor queryProjectionWeights,
            AbstractTensor keyProjectionWeights,
            AbstractTensor valueProjectionWeights,
            AbstractTensor outputProjectionWeights,
            int localAttentionLength,
            int embeddingLength) {
        if (queryProjectionWeights.shape().first() != localAttentionLength || queryProjectionWeights.shape().last() != embeddingLength) {
            throw new IllegalArgumentException("queryProjectionWeights must have shape [localAttention, embedding]");
        }
        if (keyProjectionWeights.shape().first() != localAttentionLength || keyProjectionWeights.shape().last() != embeddingLength) {
            throw new IllegalArgumentException("keyProjectionWeights must have shape [localAttention, embedding]");
        }
        if (valueProjectionWeights.shape().first() != localAttentionLength || valueProjectionWeights.shape().last() != embeddingLength) {
            throw new IllegalArgumentException("valueProjectionWeights must have shape [localAttention, embedding]");
        }
        if (outputProjectionWeights.shape().first() != embeddingLength || outputProjectionWeights.shape().last() != localAttentionLength) {
            throw new IllegalArgumentException("outputProjectionWeights must have shape [embedding, localAttention]");
        }
    }
}
