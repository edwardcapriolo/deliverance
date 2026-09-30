package io.teknek.deliverance.tensor2;

public class Transpose {
    private final TensorRef source;
    private TensorRef destination;

    public Transpose(TensorRef source) {
        this.source = source;
    }

    public Transpose destination(TensorRef destination) {
        this.destination = destination;
        return this;
    }

    public TensorRef getSource() { return source; }
    public TensorRef getDestination() { return destination; }
}
