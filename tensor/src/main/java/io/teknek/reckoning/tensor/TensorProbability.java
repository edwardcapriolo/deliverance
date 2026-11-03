package io.teknek.reckoning.tensor;

import com.google.common.base.Preconditions;
import io.teknek.reckoning.DType;
import io.teknek.reckoning.tensor.TensorShape;
import io.teknek.reckoning.tensor2.Exp;
import io.teknek.reckoning.tensor2.Lighter;
import io.teknek.reckoning.tensor2.Max;
import io.teknek.reckoning.tensor2.Sum;
import io.teknek.reckoning.tensor2.TensorRef;
import net.jafama.FastMath;

import java.util.Objects;

/** Probability-distribution helpers built by composing primitive tensor operations. */
public final class TensorProbability {
    private TensorProbability() {
    }

    /** Computes token entropy for logits shaped {@code [batch, positions, vocab]}. */
    public static void entropy(TensorRef tokenEntropy, TensorRef logits, Lighter lighter) {
        Objects.requireNonNull(tokenEntropy, "tokenEntropy");
        Objects.requireNonNull(logits, "logits");
        Objects.requireNonNull(lighter, "lighter");
        Preconditions.checkArgument(logits.dims() == 3, "logits must have shape [batch, positions, vocab]");
        Preconditions.checkArgument(tokenEntropy.dims() == 2, "tokenEntropy must have shape [batch, positions]");
        Preconditions.checkArgument(tokenEntropy.dType() == DType.F32, "tokenEntropy must be F32");
        int batchSize = logits.shape().dim(0);
        int positions = logits.shape().dim(1);
        int vocabSize = logits.shape().dim(2);
        Preconditions.checkArgument(tokenEntropy.shape().dim(0) == batchSize
                        && tokenEntropy.shape().dim(1) == positions,
                "tokenEntropy must have shape [batch, positions]");

        try (TensorRef shifted = lighter.allocate(DType.F32, TensorShape.of(positions, vocabSize));
             TensorRef exp = lighter.allocate(DType.F32, TensorShape.of(positions, vocabSize));
             TensorRef max = lighter.allocate(DType.F32, TensorShape.of(1, 1));
             TensorRef sum = lighter.allocate(DType.F32, TensorShape.of(1, 1));
             TensorRef weighted = lighter.allocate(DType.F32, TensorShape.of(1, 1))) {
            for (int batch = 0; batch < batchSize; batch++) {
                try (TensorRef batchLogits = logits.slice(batch)) {
                    for (int position = 0; position < positions; position++) {
                        lighter.max(new Max(batchLogits).into(max).row(position).offsetAndLength(0, vocabSize));
                        float maximum = max.get(0, 0);
                        for (int token = 0; token < vocabSize; token++) {
                            shifted.set(batchLogits.get(position, token) - maximum, position, token);
                        }
                    }
                }
                lighter.exp(new Exp(shifted).into(exp).offsetAndLength(0, vocabSize));
                for (int position = 0; position < positions; position++) {
                    lighter.sum(new Sum(exp).into(sum).row(position).offsetAndLength(0, vocabSize));
                    try (TensorRef expRow = exp.slice(position);
                         TensorRef shiftedRow = shifted.slice(position)) {
                        lighter.dotProductRows(weighted, expRow, shiftedRow, 0, 0, vocabSize, 0, 1, 0);
                    }
                    float sumExp = sum.get(0, 0);
                    tokenEntropy.set((float) (FastMath.log(sumExp) - weighted.get(0, 0) / sumExp), batch, position);
                }
            }
        }
    }
}
