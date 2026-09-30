package io.teknek.deliverance.tensor2;

public class ArgMax {
    private final TensorRef input;
    private TensorRef output;
    private int offset;
    private int length;

    public ArgMax(TensorRef input) {
        this.input = input;
    }

    public ArgMax into(TensorRef output) {
        this.output = output;
        return this;
    }

    public ArgMax offsetAndLength(int offset, int length) {
        this.offset = offset;
        this.length = length;
        return this;
    }

    public TensorRef getInput() { return input; }
    public TensorRef getOutput() { return output; }
    public int getOffset() { return offset; }
    public int getLength() { return length; }
}
