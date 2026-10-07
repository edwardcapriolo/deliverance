package io.teknek.deliverance.tensor2;

import java.util.Objects;

/** Paired projection request used by the gate/up native fast path. */
public final class DotProductBatchChunk {
    private TensorRef[] results;
    private TensorRef input;
    private TensorRef[] weights;
    private int inputColumnStart;
    private int weightColumnStart;
    private int columnLength;
    private int weightRowStart;
    private int weightRowCount;
    private int outputColumnStart;

    public DotProductBatchChunk results(TensorRef... results) {
        this.results = Objects.requireNonNull(results, "results").clone();
        return this;
    }

    public DotProductBatchChunk input(TensorRef input) {
        this.input = input;
        return this;
    }

    public DotProductBatchChunk weights(TensorRef... weights) {
        this.weights = Objects.requireNonNull(weights, "weights").clone();
        return this;
    }

    public DotProductBatchChunk inputColumnStart(int value) {
        inputColumnStart = value;
        return this;
    }

    public DotProductBatchChunk weightColumnStart(int value) {
        weightColumnStart = value;
        return this;
    }

    public DotProductBatchChunk columnLength(int value) {
        columnLength = value;
        return this;
    }

    public DotProductBatchChunk weightRowStart(int value) {
        weightRowStart = value;
        return this;
    }

    public DotProductBatchChunk weightRowCount(int value) {
        weightRowCount = value;
        return this;
    }

    public DotProductBatchChunk outputColumnStart(int value) {
        outputColumnStart = value;
        return this;
    }

    TensorRef[] results() { return results; }
    TensorRef input() { return input; }
    TensorRef[] weights() { return weights; }
    int inputColumnStart() { return inputColumnStart; }
    int weightColumnStart() { return weightColumnStart; }
    int columnLength() { return columnLength; }
    int weightRowStart() { return weightRowStart; }
    int weightRowCount() { return weightRowCount; }
    int outputColumnStart() { return outputColumnStart; }
}
