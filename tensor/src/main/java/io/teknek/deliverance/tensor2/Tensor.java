package io.teknek.deliverance.tensor2;

import java.lang.foreign.MemorySegment;

abstract class Tensor {
    /** Returns an opaque device handle, or {@code 0} for CPU storage. */
    long deviceHandle() {
        return 0L;
    }


    public abstract float get(int... dims) ;
    public abstract float get(int row, int column);
    public abstract void set(float v, int row, int column) ;
    public abstract void set(float v, int... dims);
    public abstract MemorySegment getMemorySegment();
    public abstract int getMemorySegmentOffset(int offset);
}
