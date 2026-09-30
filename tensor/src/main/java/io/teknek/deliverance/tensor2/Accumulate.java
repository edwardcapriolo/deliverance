package io.teknek.deliverance.tensor2;

public class Accumulate {
    private final TensorRef source;
    private TensorRef destination;
    private int offset;
    private int length;

    public Accumulate(TensorRef source) {
        this.source = source;
    }

    public Accumulate into(TensorRef destination) {
        this.destination = destination;
        return this;
    }

    public Accumulate offsetAndLength(int offset, int length) {
        this.offset = offset;
        this.length = length;
        return this;
    }

    public TensorRef getSource() { return source; }
    public TensorRef getDestination() { return destination; }
    public int getOffset() { return offset; }
    public int getLength() { return length; }
}
