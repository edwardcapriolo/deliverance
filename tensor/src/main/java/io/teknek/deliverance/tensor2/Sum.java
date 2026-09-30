package io.teknek.deliverance.tensor2;

public class Sum {
    private final TensorRef source;
    private TensorRef destination;
    private int row;
    private int offset;
    private int length;

    public Sum(TensorRef source) {
        this.source = source;
    }

    public Sum into(TensorRef destination) {
        this.destination = destination;
        return this;
    }

    public Sum row(int row) {
        this.row = row;
        return this;
    }

    public Sum offsetAndLength(int offset, int length) {
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
