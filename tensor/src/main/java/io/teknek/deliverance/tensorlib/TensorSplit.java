package io.teknek.deliverance.tensorlib;

public class TensorSplit {
    long offset;
    long length;

    public TensorSplit(long x, long y) {
        this.offset = x;
        this.length = y;
    }

    @Override
    public String toString() {
        return "TSplit{" +
                "offset=" + offset +
                ", lenth=" + length +
                '}';
    }
}
