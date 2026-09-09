package io.teknek.deliverance.tensor2;

public class MultiplyAccumulate {
    private final TensorRef source;
    private TensorRef destination;
    private int offset;
    private int length;
    public MultiplyAccumulate(TensorRef source){
        this.source = source;
    }
    public MultiplyAccumulate into(TensorRef destination){
        this.destination = destination;
        return this;
    }
    public MultiplyAccumulate offsetAndLength(int offset, int length){
        this.offset = offset;
        this.length = length;
        return this;
    }

    public TensorRef getSource() {
        return source;
    }

    public TensorRef getDestination() {
        return destination;
    }

    public void setDestination(TensorRef destination) {
        this.destination = destination;
    }

    public int getOffset() {
        return offset;
    }

    public void setOffset(int offset) {
        this.offset = offset;
    }

    public int getLength() {
        return length;
    }

    public void setLength(int length) {
        this.length = length;
    }
}
