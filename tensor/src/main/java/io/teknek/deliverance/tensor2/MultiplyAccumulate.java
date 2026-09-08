package io.teknek.deliverance.tensor2;

public class MultiplyAccumulate {
    private final TensorRef b;
    private TensorRef a;
    private int offset;
    private int length;
    public MultiplyAccumulate(TensorRef b){
        this.b = b;
    }
    public MultiplyAccumulate into(TensorRef a){
        this.a = a;
        return this;
    }
    public MultiplyAccumulate offsetAndLength(int offset, int length){
        this.offset = offset;
        this.length = length;
        return this;
    }

    public TensorRef getB() {
        return b;
    }

    public TensorRef getA() {
        return a;
    }

    public void setA(TensorRef a) {
        this.a = a;
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
