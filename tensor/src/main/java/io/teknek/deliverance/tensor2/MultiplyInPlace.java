package io.teknek.deliverance.tensor2;

public class MultiplyInPlace {
    private final float factor;
    private TensorRef target;
    private int offset;
    private int length;

    public MultiplyInPlace(float factor) {
        this.factor = factor;
    }

    public MultiplyInPlace target(TensorRef target) {
        this.target = target;
        return this;
    }

    public MultiplyInPlace offsetAndLength(int offset, int length) {
        this.offset = offset;
        this.length = length;
        return this;
    }

    public float getFactor() {
        return factor;
    }

    public TensorRef getTarget() {
        return target;
    }

    public int getOffset() {
        return offset;
    }

    public int getLength() {
        return length;
    }
}
