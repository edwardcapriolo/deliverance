package io.teknek.deliverance.tensor2;

import com.google.common.base.Preconditions;
import io.teknek.deliverance.DType;
import io.teknek.dysfx.Either;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.nio.ByteOrder;

class PanamaOps implements TensorOps {
    private static final VectorSpecies<Float> F32_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final IntVector BF16_BYTE_SHIFT_256 = IntVector.broadcast(IntVector.SPECIES_256, 16);

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

    @Override
    public Either<OpSupport, Void> scale(float factor, TensorRef target, int offset, int length) {
        Preconditions.checkArgument(offset >= 0 && length >= 0 && offset + length <= target.shape().last());
        if (target.dType() == DType.F32) {
            scaleF32(factor, target, offset, length);
            return Either.Right(null);
        }
        if (target.dType() == DType.BF16) {
            scaleBF16(factor, target, offset, length);
            return Either.Right(null);
        }
        return Either.Left(OpSupport.Unsupported);
    }

    private void scaleF32(float factor, TensorRef target, int offset, int length) {
        Tensor tensor = target.underlying();
        FloatVector scale = FloatVector.broadcast(F32_SPECIES, factor);
        int end = offset + length;
        for (int row = 0; row < target.shape().first(); row++) {
            int column = offset;
            int upperBound = offset + F32_SPECIES.loopBound(length);
            for (; column < upperBound; column += F32_SPECIES.length()) {
                long targetOffset = memoryOffset(target, row, column);
                FloatVector values = FloatVector.fromMemorySegment(F32_SPECIES, tensor.getMemorySegment(), targetOffset,
                        ByteOrder.LITTLE_ENDIAN);
                values.mul(scale).intoMemorySegment(tensor.getMemorySegment(), targetOffset, ByteOrder.LITTLE_ENDIAN);
            }
            if (column < end) {
                VectorMask<Float> mask = F32_SPECIES.indexInRange(column, end);
                long targetOffset = memoryOffset(target, row, column);
                FloatVector values = FloatVector.fromMemorySegment(F32_SPECIES, tensor.getMemorySegment(), targetOffset,
                        ByteOrder.LITTLE_ENDIAN, mask);
                values.mul(scale).intoMemorySegment(tensor.getMemorySegment(), targetOffset, ByteOrder.LITTLE_ENDIAN,
                        mask);
            }
        }
    }

    private void scaleBF16(float factor, TensorRef target, int offset, int length) {
        Tensor tensor = target.underlying();
        FloatVector scale = FloatVector.broadcast(FloatVector.SPECIES_256, factor);
        int end = offset + length;
        for (int row = 0; row < target.shape().first(); row++) {
            int column = offset;
            int upperBound = offset + FloatVector.SPECIES_256.loopBound(length);
            for (; column < upperBound; column += FloatVector.SPECIES_256.length()) {
                long targetOffset = memoryOffset(target, row, column);
                var values = ShortVector.fromMemorySegment(ShortVector.SPECIES_128, tensor.getMemorySegment(),
                                targetOffset, ByteOrder.LITTLE_ENDIAN)
                        .convertShape(VectorOperators.S2I, IntVector.SPECIES_256, 0)
                        .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_256)
                        .reinterpretAsFloats();
                var result = values.mul(scale)
                        .reinterpretAsInts()
                        .lanewise(VectorOperators.ASHR, BF16_BYTE_SHIFT_256)
                        .convertShape(VectorOperators.I2S, ShortVector.SPECIES_128, 0);
                ((ShortVector) result).intoMemorySegment(tensor.getMemorySegment(), targetOffset,
                        ByteOrder.LITTLE_ENDIAN);
            }
            for (; column < end; column++) {
                target.underlying().set(target.underlying().get(row, column) * factor, row, column);
            }
        }
    }

    private static long memoryOffset(TensorRef tensor, int row, int column) {
        return tensor.underlying().getMemorySegmentOffset(tensor.shape().getOffset(row, column));
    }
}
