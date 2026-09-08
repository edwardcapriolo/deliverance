package io.teknek.deliverance.tensor2;

import java.lang.foreign.MemorySegment;

abstract class Tensor {

    public abstract float get(int... dims) ;
    public abstract float get(int row, int column);
    public abstract void set(float v, int row, int column) ;
    public abstract void set(float v, int... dims);
    public abstract MemorySegment getMemorySegment();
    public abstract int getMemorySegmentOffset(int offset);
}
