package io.teknek.deliverance.tensor2;

import com.google.common.base.Preconditions;
import io.teknek.deliverance.DType;
import io.teknek.dysfx.Either;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorSpecies;

import java.nio.ByteOrder;

class PanamaOps implements TensorOps {
    private static final VectorSpecies<Float> F32_SPECIES = FloatVector.SPECIES_PREFERRED;

    @Override
    public Either<OpSupport, Void> multiplyAccumulate(TensorRef a, TensorRef b, int offset, int length) {
        if (a.dType() != DType.F32 || b.dType() != DType.F32) {
            return Either.Left(OpSupport.Unsupported);
        }
        Tensor at = a.underlying();
        Tensor bt = b.underlying();

        Preconditions.checkArgument(a.dims() == b.dims());
        Preconditions.checkArgument(a.shape().last() == b.shape().last());
        Preconditions.checkArgument(b.shape().first() == 1 || a.shape().first() == b.shape().first());
        Preconditions.checkArgument(offset >= 0 && length >= 0 && offset + length <= a.shape().last());

        boolean isBatch = b.shape().first() > 1;
        int end = offset + length;
        for (int ai = 0; ai < a.shape().first(); ai++) {
            int bi = isBatch ? ai : 0;
            int i = offset;
            int upperBound = offset + F32_SPECIES.loopBound(length);
            for (; i < upperBound; i += F32_SPECIES.length()) {
                long aOffset = memoryOffset(a, ai, i);
                long bOffset = memoryOffset(b, bi, i);
                FloatVector va = FloatVector.fromMemorySegment(F32_SPECIES, at.getMemorySegment(), aOffset,
                        ByteOrder.LITTLE_ENDIAN);
                FloatVector vb = FloatVector.fromMemorySegment(F32_SPECIES, bt.getMemorySegment(), bOffset,
                        ByteOrder.LITTLE_ENDIAN);
                va.mul(vb).intoMemorySegment(at.getMemorySegment(), aOffset, ByteOrder.LITTLE_ENDIAN);
            }
            if (i < end) {
                VectorMask<Float> mask = F32_SPECIES.indexInRange(i, end);
                long aOffset = memoryOffset(a, ai, i);
                long bOffset = memoryOffset(b, bi, i);
                FloatVector va = FloatVector.fromMemorySegment(F32_SPECIES, at.getMemorySegment(), aOffset,
                        ByteOrder.LITTLE_ENDIAN, mask);
                FloatVector vb = FloatVector.fromMemorySegment(F32_SPECIES, bt.getMemorySegment(), bOffset,
                        ByteOrder.LITTLE_ENDIAN, mask);
                va.mul(vb).intoMemorySegment(at.getMemorySegment(), aOffset, ByteOrder.LITTLE_ENDIAN, mask);
            }
        }
        return Either.Right(null);
    }

    private static long memoryOffset(TensorRef tensor, int row, int column) {
        return tensor.underlying().getMemorySegmentOffset(tensor.shape().getOffset(row, column));
    }
}
