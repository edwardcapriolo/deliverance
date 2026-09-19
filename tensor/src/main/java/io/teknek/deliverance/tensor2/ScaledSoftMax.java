package io.teknek.deliverance.tensor2;

public class ScaledSoftMax {
    private final float scale;
    private TensorRef target;
    private int offset;
    private int length;
    private Float softcap;

    public ScaledSoftMax(float scale) {
        this.scale = scale;
    }

    public ScaledSoftMax target(TensorRef target) {
        this.target = target;
        return this;
    }

    public ScaledSoftMax offsetAndLength(int offset, int length) {
        this.offset = offset;
        this.length = length;
        return this;
    }

    public ScaledSoftMax softcap(Float softcap) {
        this.softcap = softcap;
        return this;
    }

    public float getScale() {
        return scale;
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

    public Float getSoftcap() {
        return softcap;
    }
}
