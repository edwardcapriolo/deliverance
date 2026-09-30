package io.teknek.deliverance.tensor2;

public class Exp {
    private final TensorRef source;
    private TensorRef destination;
    private int offset;
    private int length;

    public Exp(TensorRef source) {
        this.source = source;
    }

    public Exp into(TensorRef destination) {
        this.destination = destination;
        return this;
    }

    public Exp offsetAndLength(int offset, int length) {
        this.offset = offset;
        this.length = length;
        return this;
    }

    public TensorRef getSource() { return source; }
    public TensorRef getDestination() { return destination; }
    public int getOffset() { return offset; }
    public int getLength() { return length; }
}
