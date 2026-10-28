package io.teknek.deliverance.model.bert;

import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;

import java.util.function.Function;

/** Tensor2 implementation of the RoBERTa sentence-classification head. */
public final class RobertaClassificationHead {
    private final TensorRef denseWeight;
    private final TensorRef denseBias;
    private final TensorRef outputWeight;
    private final TensorRef outputBias;
    private final int hiddenSize;
    private final int labelCount;

    public RobertaClassificationHead(TensorRef denseWeight, TensorRef denseBias,
            TensorRef outputWeight, TensorRef outputBias, int hiddenSize, int labelCount) {
        this.denseWeight = denseWeight;
        this.denseBias = denseBias;
        this.outputWeight = outputWeight;
        this.outputBias = outputBias;
        this.hiddenSize = hiddenSize;
        this.labelCount = labelCount;
    }

    /**
     * Applies the inference-time head. Dropout is intentionally omitted because models
     * are evaluated in inference mode.
     */
    public TensorRef forward(TensorRef sequenceOutput, Lighter lighter,
            Function<TensorShape, TensorRef> allocator) {
        try (TensorRef firstToken = sequenceOutput.slice(0);
             TensorRef dense = allocator.apply(TensorShape.of(1, hiddenSize))) {
            lighter.dotProductRows(dense, firstToken, denseWeight, 0, hiddenSize, 0, hiddenSize, 0);
            for (int column = 0; column < hiddenSize; column++) {
                dense.set((float) Math.tanh(dense.get(0, column) + denseBias.get(0, column)), 0, column);
            }
            TensorRef logits = allocator.apply(TensorShape.of(1, labelCount));
            lighter.dotProductRows(logits, dense, outputWeight, 0, hiddenSize, 0, labelCount, 0);
            for (int label = 0; label < labelCount; label++) {
                logits.set(logits.get(0, label) + outputBias.get(0, label), 0, label);
            }
            return logits;
        }
    }
}
