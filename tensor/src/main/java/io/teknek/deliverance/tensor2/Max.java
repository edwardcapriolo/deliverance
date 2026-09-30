package io.teknek.deliverance.tensor2;

public class Max {
    private final TensorRef source;
    private TensorRef destination;
    private int row;
    private int offset;
    private int length;

    public Max(TensorRef source) {
        this.source = source;
    }

    public Max into(TensorRef destination) {
        this.destination = destination;
        return this;
    }

    public Max row(int row) {
        this.row = row;
        return this;
    }

    public Max offsetAndLength(int offset, int length) {
        this.offset = offset;
        this.length = length;
        return this;
    }

    public TensorRef getSource() { return source; }
    public TensorRef getDestination() { return destination; }
    public int getRow() { return row; }
    public int getOffset() { return offset; }
    public int getLength() { return length; }
}
