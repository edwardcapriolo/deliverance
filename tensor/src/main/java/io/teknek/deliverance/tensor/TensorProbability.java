package io.teknek.deliverance.tensor;

import com.google.common.base.Preconditions;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.operations.TensorOperations;
import net.jafama.FastMath;

import java.util.Objects;

/** Probability-distribution helpers built by composing primitive tensor operations. */
public final class TensorProbability {
    private TensorProbability() {
    }

    /**
     * Computes token entropy for logits shaped {@code [batch, positions, vocab]}.
     *
     * <p>{@code tokenEntropy} must have shape {@code [batch, positions]}. Each output entry is the entropy of the
     * corresponding vocabulary row after softmax. The implementation composes primitive provider operations rather than
     * adding a model-specific provider method: row max, exp, row sum, and dot product are stitched together here.</p>
     */
    public static void entropy(AbstractTensor tokenEntropy, AbstractTensor logits, TensorOperations tensorOperations) {
        Objects.requireNonNull(tokenEntropy, "tokenEntropy");
        Objects.requireNonNull(logits, "logits");
        Objects.requireNonNull(tensorOperations, "tensorOperations");
        TensorMutability.requireWritable(tokenEntropy, "entropy");
        Preconditions.checkArgument(logits.dims() == 3, "logits must have shape [batch, positions, vocab]");
        Preconditions.checkArgument(tokenEntropy.dims() == 2, "tokenEntropy must have shape [batch, positions]");
        Preconditions.checkArgument(tokenEntropy.dType() == DType.F32, "tokenEntropy must be F32");
        int batchSize = logits.shape().dim(0);
        int positions = logits.shape().dim(1);
        int vocabSize = logits.shape().dim(2);
        Preconditions.checkArgument(tokenEntropy.shape().dim(0) == batchSize
                        && tokenEntropy.shape().dim(1) == positions,
                "tokenEntropy must have shape [batch, positions]");

        try (FloatBufferTensor shifted = new FloatBufferTensor(positions, vocabSize);
             FloatBufferTensor exp = new FloatBufferTensor(positions, vocabSize)) {
            for (int batch = 0; batch < batchSize; batch++) {
                try (AbstractTensor batchLogits = logits.slice(batch)) {
                    for (int position = 0; position < positions; position++) {
                        float max = tensorOperations.max(batchLogits, position, 0, vocabSize);
                        for (int token = 0; token < vocabSize; token++) {
                            shifted.set(batchLogits.get(position, token) - max, position, token);
                        }
                    }
                }
                tensorOperations.exp(shifted, exp, 0, vocabSize);
                for (int position = 0; position < positions; position++) {
                    float sumExp = tensorOperations.sum(exp, position, 0, vocabSize);
                    float weighted;
                    try (AbstractTensor expRow = exp.slice(position);
                         AbstractTensor shiftedRow = shifted.slice(position)) {
                        weighted = tensorOperations.dotProduct(expRow, shiftedRow, 0, 0, vocabSize);
                    }
                    tokenEntropy.set((float) (FastMath.log(sumExp) - weighted / sumExp), batch, position);
                }
            }
        }
    }
}
