package io.teknek.deliverance.tensor2;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.ActivationFunction;

/** Fused activation, elementwise multiply, and quantization request. */
public final class ActivationMultiplyQuantize {
    private final TensorRef gate;
    private final TensorRef up;
    private final ActivationFunction.Type activation;
    private final DType qtype;
    private int offset;
    private int length;
    private TensorRef output;

    public ActivationMultiplyQuantize(TensorRef gate, TensorRef up, ActivationFunction.Type activation, DType qtype) {
        this.gate = gate;
        this.up = up;
        this.activation = activation;
        this.qtype = qtype;
        this.offset = 0;
        this.length = 0;
    }

    public ActivationMultiplyQuantize output(TensorRef output) {
        this.output = output;
        return this;
    }

    public ActivationMultiplyQuantize offsetAndLength(int offset, int length) {
        this.offset = offset;
        this.length = length;
        return this;
    }

    TensorRef gate() { return gate; }
    TensorRef up() { return up; }
    TensorRef output() { return output; }
    ActivationFunction.Type activation() { return activation; }
    DType qtype() { return qtype; }
    int offset() { return offset; }
    int length() { return length; }
}
