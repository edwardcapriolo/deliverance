package io.teknek.deliverance.tensor2;

import com.google.common.base.Preconditions;
import io.teknek.deliverance.DType;
import io.teknek.dysfx.Either;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.nio.ByteOrder;

class PanamaOps implements TensorOps {
    private static final VectorSpecies<Float> F32_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Byte> Q8_BYTE_SPECIES = ByteVector.SPECIES_128;
    private static final VectorSpecies<Float> Q8_FLOAT_SPECIES = FloatVector.SPECIES_512;
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
    public Either<OpSupport, Void> batchDotProduct(BatchDotProduct operation) {
        TensorRef result = operation.result();
        TensorRef a = operation.a();
        TensorRef b = operation.b();
        if (result.dType() != DType.F32 || a.dType() != DType.F32) {
            return Either.Left(OpSupport.Unsupported);
        }
        TensorRef q8Scale = Q8Layout.scale(b);
        if (b.dType() == DType.I8 && q8Scale != null && q8Aligned(operation)) {
            new F32Q8BatchDotProductGemmer(operation, q8Scale).matmul();
            return Either.Right(null);
        }
        if (b.dType() != DType.F32) {
            return Either.Left(OpSupport.Unsupported);
        }
        new F32BatchDotProductGemmer(operation).matmul();
        return Either.Right(null);
    }

    private static boolean q8Aligned(BatchDotProduct operation) {
        return operation.aColumnOffset() % Q8Layout.BLOCK_SIZE == 0
                && operation.bColumnOffset() % Q8Layout.BLOCK_SIZE == 0
                && operation.columnLength() % Q8Layout.BLOCK_SIZE == 0;
    }

    private static final class F32BatchDotProductGemmer {
        private final BatchDotProduct operation;

        private F32BatchDotProductGemmer(BatchDotProduct operation) {
            this.operation = operation;
        }

        private void matmul() {
            TensorRef result = operation.result();
            TensorRef a = operation.a();
            TensorRef b = operation.b();
            Tensor resultTensor = result.underlying();
            Tensor aTensor = a.underlying();
            Tensor bTensor = b.underlying();
            int bEnd = operation.bRowOffset() + operation.rowChunkSize();
            for (int resultRow = 0; resultRow < result.shape().first(); resultRow++) {
                int aRow = operation.aRowOffset() + resultRow;
                for (int bRow = operation.bRowOffset(); bRow < bEnd; bRow++) {
                    resultTensor.set(dot(a, b, aTensor, bTensor, aRow, bRow), resultRow,
                            bRow + operation.resultRowOffset());
                }
            }
        }

        private float dot(TensorRef a, TensorRef b, Tensor aTensor, Tensor bTensor, int aRow, int bRow) {
            FloatVector acc = FloatVector.zero(F32_SPECIES);
            int k = 0;
            int upperBound = F32_SPECIES.loopBound(operation.columnLength());
            for (; k < upperBound; k += F32_SPECIES.length()) {
                FloatVector av = FloatVector.fromMemorySegment(F32_SPECIES, aTensor.getMemorySegment(),
                        memoryOffset(a, aRow, operation.aColumnOffset() + k), ByteOrder.LITTLE_ENDIAN);
                FloatVector bv = FloatVector.fromMemorySegment(F32_SPECIES, bTensor.getMemorySegment(),
                        memoryOffset(b, bRow, operation.bColumnOffset() + k), ByteOrder.LITTLE_ENDIAN);
                acc = av.fma(bv, acc);
            }
            float sum = acc.reduceLanes(VectorOperators.ADD);
            for (; k < operation.columnLength(); k++) {
                sum += aTensor.get(aRow, operation.aColumnOffset() + k)
                        * bTensor.get(bRow, operation.bColumnOffset() + k);
            }
            return sum;
        }
    }

    private static final class F32Q8BatchDotProductGemmer {
        private final BatchDotProduct operation;
        private final TensorRef q8Scale;

        private F32Q8BatchDotProductGemmer(BatchDotProduct operation, TensorRef q8Scale) {
            this.operation = operation;
            this.q8Scale = q8Scale;
        }

        private void matmul() {
            TensorRef result = operation.result();
            TensorRef a = operation.a();
            TensorRef b = operation.b();
            Tensor resultTensor = result.underlying();
            Tensor aTensor = a.underlying();
            Tensor bTensor = b.underlying();
            int bEnd = operation.bRowOffset() + operation.rowChunkSize();
            for (int resultRow = 0; resultRow < result.shape().first(); resultRow++) {
                int aRow = operation.aRowOffset() + resultRow;
                for (int bRow = operation.bRowOffset(); bRow < bEnd; bRow++) {
                    resultTensor.set(dot(a, b, aTensor, bTensor, aRow, bRow), resultRow,
                            bRow + operation.resultRowOffset());
                }
            }
        }

        private float dot(TensorRef a, TensorRef b, Tensor aTensor, Tensor bTensor, int aRow, int bRow) {
            FloatVector acc0 = FloatVector.zero(Q8_FLOAT_SPECIES);
            FloatVector acc1 = FloatVector.zero(Q8_FLOAT_SPECIES);
            int aColumn = operation.aColumnOffset();
            int bColumn = operation.bColumnOffset();
            int end = operation.aColumnOffset() + operation.columnLength();
            for (; aColumn < end; aColumn += Q8Layout.BLOCK_SIZE, bColumn += Q8Layout.BLOCK_SIZE) {
                FloatVector scale = FloatVector.broadcast(Q8_FLOAT_SPECIES,
                        q8Scale.underlying().get(bRow, Q8Layout.scaleColumn(bColumn)));
                long bOffset0 = memoryOffset(b, bRow, bColumn);
                long bOffset1 = memoryOffset(b, bRow, bColumn + Q8Layout.BLOCK_SIZE / 2);
                ByteVector bytes0 = ByteVector.fromMemorySegment(Q8_BYTE_SPECIES, bTensor.getMemorySegment(),
                        bOffset0, ByteOrder.LITTLE_ENDIAN);
                ByteVector bytes1 = ByteVector.fromMemorySegment(Q8_BYTE_SPECIES, bTensor.getMemorySegment(),
                        bOffset1, ByteOrder.LITTLE_ENDIAN);
                FloatVector bv0 = ((FloatVector) bytes0.convertShape(VectorOperators.B2F, Q8_FLOAT_SPECIES, 0))
                        .mul(scale);
                FloatVector bv1 = ((FloatVector) bytes1.convertShape(VectorOperators.B2F, Q8_FLOAT_SPECIES, 0))
                        .mul(scale);
                FloatVector av0 = FloatVector.fromMemorySegment(Q8_FLOAT_SPECIES, aTensor.getMemorySegment(),
                        memoryOffset(a, aRow, aColumn), ByteOrder.LITTLE_ENDIAN);
                FloatVector av1 = FloatVector.fromMemorySegment(Q8_FLOAT_SPECIES, aTensor.getMemorySegment(),
                        memoryOffset(a, aRow, aColumn + Q8Layout.BLOCK_SIZE / 2), ByteOrder.LITTLE_ENDIAN);
                acc0 = av0.fma(bv0, acc0);
                acc1 = av1.fma(bv1, acc1);
            }
            return acc0.add(acc1).reduceLanes(VectorOperators.ADD);
        }
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
