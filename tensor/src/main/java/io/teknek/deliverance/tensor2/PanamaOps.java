package io.teknek.deliverance.tensor2;

// DO NOT IMPLEMENT JAVA LOOPS IN THIS PROVIDER PERIOD

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
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

class PanamaOps implements TensorOps {
    @Override
    public Either<OpSupport, Void> argMax(TensorRef input, TensorRef output, int offset, int length) {
        if (input.dType() != DType.F32) {
            return Either.Left(OpSupport.Unsupported);
        }
        Preconditions.checkArgument(input.shape().first() == 1, "argMax expects one row");
        Preconditions.checkArgument(output.shape().first() == 1 && output.shape().last() == 2,
                "argMax output must have shape [1, 2]");
        Preconditions.checkArgument(output.dType() == DType.F32, "argMax output must be F32");
        Preconditions.checkArgument(offset >= 0 && length > 0 && offset + length <= input.shape().last(),
                "argMax window out of bounds");
        int i = offset;
        int upperBound = offset + F32_SPECIES.loopBound(length);
        FloatVector maxVector = FloatVector.broadcast(F32_SPECIES, Float.NEGATIVE_INFINITY);
        for (; i < upperBound; i += F32_SPECIES.length()) {
            maxVector = maxVector.max(FloatVector.fromMemorySegment(F32_SPECIES, input.memorySegment(),
                    memoryOffset(input, 0, i), ByteOrder.LITTLE_ENDIAN));
        }
        float maxValue = maxVector.reduceLanes(VectorOperators.MAX);
        for (; i < offset + length; i++) {
            maxValue = Math.max(maxValue, input.get(0, i));
        }
        int maxIndex = offset;
        for (i = offset; i < offset + length; i++) {
            if (input.get(0, i) == maxValue) {
                maxIndex = i;
                break;
            }
        }
        output.set(maxIndex, 0, 0);
        output.set(maxValue, 0, 1);
        return Either.Right(null);
    }

    @Override
    public Either<OpSupport, Void> accumulate(TensorRef a, TensorRef b, int offset, int length) {
        boolean supported = a.dType() == DType.F32
                ? b.dType() == DType.F32 || b.dType() == DType.Q4 || b.dType() == DType.BF16
                : a.dType() == DType.BF16 && b.dType() == DType.BF16;
        if (!supported) {
            return Either.Left(OpSupport.Unsupported);
        }
        Preconditions.checkArgument(a.dims() == b.dims());
        Preconditions.checkArgument(a.shape().last() == b.shape().last());
        Preconditions.checkArgument(b.shape().first() == 1 || a.shape().first() == b.shape().first());
        Preconditions.checkArgument(offset >= 0 && length >= 0 && offset + length <= a.shape().last());
        boolean broadcast = b.shape().first() == 1;
        for (int row = 0; row < a.shape().first(); row++) {
            int sourceRow = broadcast ? 0 : row;
            if (a.dType() == DType.F32 && b.dType() == DType.F32) {
                accumulateF32(a, b, row, sourceRow, offset, length);
            } else if (a.dType() == DType.F32 && b.dType() == DType.BF16) {
                accumulateF32Bf16(a, b, row, sourceRow, offset, length);
            } else if (a.dType() == DType.BF16 && b.dType() == DType.BF16) {
                accumulateBf16(a, b, row, sourceRow, offset, length);
            } else {
                accumulateF32Q4(a, b, row, sourceRow, offset, length);
            }
        }
        return Either.Right(null);
    }

    private void accumulateF32(TensorRef target, TensorRef source, int targetRow, int sourceRow,
            int offset, int length) {
        int i = offset;
        int end = offset + length;
        int upperBound = offset + F32_SPECIES.loopBound(length);
        for (; i < upperBound; i += F32_SPECIES.length()) {
            FloatVector a = FloatVector.fromMemorySegment(F32_SPECIES, target.memorySegment(),
                    memoryOffset(target, targetRow, i), ByteOrder.LITTLE_ENDIAN);
            FloatVector b = FloatVector.fromMemorySegment(F32_SPECIES, source.memorySegment(),
                    memoryOffset(source, sourceRow, i), ByteOrder.LITTLE_ENDIAN);
            a.add(b).intoMemorySegment(target.memorySegment(), memoryOffset(target, targetRow, i),
                    ByteOrder.LITTLE_ENDIAN);
        }
        for (; i < end; i++) {
            target.set(target.get(targetRow, i) + source.get(sourceRow, i), targetRow, i);
        }
    }

    private void accumulateF32Bf16(TensorRef target, TensorRef source, int targetRow, int sourceRow,
            int offset, int length) {
        int i = offset;
        int end = offset + length;
        int upperBound = offset + F32_BF16_SPECIES.loopBound(length);
        for (; i < upperBound; i += F32_BF16_SPECIES.length()) {
            FloatVector a = FloatVector.fromMemorySegment(F32_BF16_SPECIES, target.memorySegment(),
                    memoryOffset(target, targetRow, i), ByteOrder.LITTLE_ENDIAN);
            a.add(bf16Vector(source, sourceRow, i)).intoMemorySegment(target.memorySegment(),
                    memoryOffset(target, targetRow, i), ByteOrder.LITTLE_ENDIAN);
        }
        for (; i < end; i++) {
            target.set(target.get(targetRow, i) + source.get(sourceRow, i), targetRow, i);
        }
    }

    private void accumulateBf16(TensorRef target, TensorRef source, int targetRow, int sourceRow,
            int offset, int length) {
        int i = offset;
        int end = offset + length;
        int upperBound = offset + F32_BF16_SPECIES.loopBound(length);
        for (; i < upperBound; i += F32_BF16_SPECIES.length()) {
            FloatVector sum = bf16Vector(target, targetRow, i).add(bf16Vector(source, sourceRow, i));
            IntVector rounded = roundFloatBitsToBf16(sum.reinterpretAsInts());
            ShortVector result = (ShortVector) rounded.lanewise(VectorOperators.LSHR, 16)
                    .convertShape(VectorOperators.I2S, BF16_SPECIES, 0);
            result.intoMemorySegment(target.memorySegment(), memoryOffset(target, targetRow, i),
                    ByteOrder.LITTLE_ENDIAN);
        }
        for (; i < end; i++) {
            target.set(target.get(targetRow, i) + source.get(sourceRow, i), targetRow, i);
        }
    }

    private void accumulateF32Q4(TensorRef target, TensorRef source, int targetRow, int sourceRow,
            int offset, int length) {
        int end = offset + length;
        for (int column = offset; column < end; column += Q4Layout.BLOCK_SIZE) {
            int blockLength = Math.min(Q4Layout.BLOCK_SIZE, end - column);
            if (blockLength != Q4Layout.BLOCK_SIZE) {
                for (int i = column; i < end; i++) {
                    target.set(target.get(targetRow, i) + source.get(sourceRow, i), targetRow, i);
                }
                return;
            }
            float scale = Q4Layout.scale(source).get(sourceRow, Q4Layout.blockIndex(column));
            ByteVector packed = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, source.memorySegment(),
                    memoryOffset(source, sourceRow, column), ByteOrder.LITTLE_ENDIAN);
            FloatVector low = (FloatVector) packed.lanewise(VectorOperators.AND, Q4_MASK)
                    .sub(Q4_ZERO).convertShape(VectorOperators.B2F, FloatVector.SPECIES_512, 0);
            FloatVector high = (FloatVector) packed.lanewise(VectorOperators.ASHR, Q4_SHIFT)
                    .lanewise(VectorOperators.AND, Q4_MASK).sub(Q4_ZERO)
                    .convertShape(VectorOperators.B2F, FloatVector.SPECIES_512, 0);
            FloatVector lowTarget = FloatVector.fromMemorySegment(FloatVector.SPECIES_512, target.memorySegment(),
                    memoryOffset(target, targetRow, column), ByteOrder.LITTLE_ENDIAN);
            FloatVector highTarget = FloatVector.fromMemorySegment(FloatVector.SPECIES_512, target.memorySegment(),
                    memoryOffset(target, targetRow, column + Q4Layout.HALF_BLOCK), ByteOrder.LITTLE_ENDIAN);
            lowTarget.add(low.mul(scale)).intoMemorySegment(target.memorySegment(),
                    memoryOffset(target, targetRow, column), ByteOrder.LITTLE_ENDIAN);
            highTarget.add(high.mul(scale)).intoMemorySegment(target.memorySegment(),
                    memoryOffset(target, targetRow, column + Q4Layout.HALF_BLOCK), ByteOrder.LITTLE_ENDIAN);
        }
    }
    private static final VectorSpecies<Float> F32_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Float> F32_BF16_SPECIES = FloatVector.SPECIES_256;
    private static final VectorSpecies<Integer> F16_INT_SPECIES = IntVector.SPECIES_256;
    private static final VectorSpecies<Short> BF16_SPECIES = ShortVector.SPECIES_128;
    private static final VectorSpecies<Byte> Q8_BYTE_SPECIES = ByteVector.SPECIES_128;
    private static final VectorSpecies<Float> Q8_FLOAT_SPECIES = FloatVector.SPECIES_512;
    private static final ByteVector Q4_MASK = ByteVector.broadcast(ByteVector.SPECIES_128, (byte) 0x0f);
    private static final ByteVector Q4_ZERO = ByteVector.broadcast(ByteVector.SPECIES_128, (byte) 8);
    private static final ByteVector Q4_SHIFT = ByteVector.broadcast(ByteVector.SPECIES_128, (byte) 4);
    private static final IntVector BF16_BYTE_SHIFT_256 = IntVector.broadcast(IntVector.SPECIES_256, 16);
    private static final IntVector BF16_ROUND_BIAS = IntVector.broadcast(IntVector.SPECIES_256, 0x7fff);
    private static final IntVector BF16_ROUND_LSB_MASK = IntVector.broadcast(IntVector.SPECIES_256, 1);

    @Override
    public Either<OpSupport, Void> reshape(TensorRef input, TensorRef output) {
        Preconditions.checkArgument(input.dims() == 2 && output.dims() == 2, "Reshape requires 2D tensors");
        Preconditions.checkArgument(input.shape().equals(output.shape()), "Input and output shapes must match");
        if (input.dType() == output.dType()) {
            copySameDType(input, output);
            return Either.Right(null);
        }
        if (input.dType() == DType.F32 && output.dType() == DType.BF16) {
            reshapeF32ToBF16(input, output);
            return Either.Right(null);
        }
        if (input.dType() == DType.BF16 && output.dType() == DType.F32) {
            reshapeBF16ToF32(input, output);
            return Either.Right(null);
        }
        if (input.dType() == DType.F16 && output.dType() == DType.F32) {
            reshapeF16ToF32(input, output);
            return Either.Right(null);
        }
        if (input.dType() == DType.F16 && output.dType() == DType.BF16) {
            reshapeF16ToBF16(input, output);
            return Either.Right(null);
        }
        if (input.dType() == DType.F32 && output.dType() == DType.F16) {
            reshapeF32ToF16(input, output);
            return Either.Right(null);
        }
        if (input.dType() == DType.BF16 && output.dType() == DType.F16) {
            reshapeBF16ToF16(input, output);
            return Either.Right(null);
        }
        if (input.dType() == DType.F16) {
            return Either.Left(OpSupport.Unsupported);
        }
        if (output.dType() == DType.I8) {
            reshapeToI8(input, output);
            return Either.Right(null);
        }
        if (output.dType() == DType.Q4) {
            reshapeToQ4(input, output);
            return Either.Right(null);
        }
        if ((input.dType() == DType.I8 || input.dType() == DType.Q4)
                && (output.dType() == DType.F32 || output.dType() == DType.BF16)) {
            reshapeQuantizedToDense(input, output);
            return Either.Right(null);
        }
        return Either.Left(OpSupport.Unsupported);
    }

    private void copySameDType(TensorRef input, TensorRef output) {
        long bytes = physicalBytes(input);
        output.underlying().getMemorySegment().asSlice(output.memorySegmentOffset(output.shape().getOffset(0, 0)), bytes)
                .copyFrom(input.underlying().getMemorySegment().asSlice(
                        input.memorySegmentOffset(input.shape().getOffset(0, 0)), bytes));
        if (input.dType() == DType.I8) {
            copySidecar(Q8Layout.scale(input), Q8Layout.scale(output));
        } else if (input.dType() == DType.Q4) {
            copySidecar(Q4Layout.scale(input), Q4Layout.scale(output));
        }
    }

    private void copySidecar(TensorRef input, TensorRef output) {
        Preconditions.checkArgument(input != null && output != null, "Quantized tensors require scale sidecars");
        long bytes = input.shape().size() * DType.F32.size();
        output.underlying().getMemorySegment().asSlice(output.memorySegmentOffset(output.shape().getOffset(0, 0)), bytes)
                .copyFrom(input.underlying().getMemorySegment().asSlice(
                        input.memorySegmentOffset(input.shape().getOffset(0, 0)), bytes));
    }

    private long physicalBytes(TensorRef tensor) {
        return switch (tensor.dType()) {
            case F32, F16, BF16, I8 -> tensor.shape().size() * tensor.dType().size();
            case Q4 -> tensor.shape().size() / 2;
            default -> throw new IllegalArgumentException("Unsupported dtype " + tensor.dType());
        };
    }

    private void reshapeF32ToBF16(TensorRef input, TensorRef output) {
        Tensor inputTensor = input.underlying();
        Tensor outputTensor = output.underlying();
        for (int row = 0; row < input.shape().first(); row++) {
            int column = 0;
            int upperBound = F32_BF16_SPECIES.loopBound(input.shape().last());
            for (; column < upperBound; column += F32_BF16_SPECIES.length()) {
                FloatVector values = FloatVector.fromMemorySegment(F32_BF16_SPECIES, inputTensor.getMemorySegment(),
                        memoryOffset(input, row, column), ByteOrder.LITTLE_ENDIAN);
                IntVector bits = values.reinterpretAsInts();
                IntVector rounded = bits.add(BF16_ROUND_BIAS)
                        .add(bits.lanewise(VectorOperators.LSHR, BF16_BYTE_SHIFT_256)
                                .lanewise(VectorOperators.AND, BF16_ROUND_LSB_MASK));
                var bf16 = rounded.lanewise(VectorOperators.LSHR, BF16_BYTE_SHIFT_256)
                        .convertShape(VectorOperators.I2S, BF16_SPECIES, 0);
                ((ShortVector) bf16).intoMemorySegment(outputTensor.getMemorySegment(), memoryOffset(output, row, column),
                        ByteOrder.LITTLE_ENDIAN);
            }
            for (; column < input.shape().last(); column++) {
                outputTensor.set(inputTensor.get(row, column), row, column);
            }
        }
    }

    private void reshapeBF16ToF32(TensorRef input, TensorRef output) {
        Tensor inputTensor = input.underlying();
        Tensor outputTensor = output.underlying();
        for (int row = 0; row < input.shape().first(); row++) {
            int column = 0;
            int upperBound = F32_BF16_SPECIES.loopBound(input.shape().last());
            for (; column < upperBound; column += F32_BF16_SPECIES.length()) {
                var values = ShortVector.fromMemorySegment(BF16_SPECIES, inputTensor.getMemorySegment(),
                                memoryOffset(input, row, column), ByteOrder.LITTLE_ENDIAN)
                        .convertShape(VectorOperators.S2I, IntVector.SPECIES_256, 0)
                        .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_256)
                        .reinterpretAsFloats();
                values.intoMemorySegment(outputTensor.getMemorySegment(), memoryOffset(output, row, column),
                        ByteOrder.LITTLE_ENDIAN);
            }
            for (; column < input.shape().last(); column++) {
                outputTensor.set(inputTensor.get(row, column), row, column);
            }
        }
    }

    private void reshapeF16ToF32(TensorRef input, TensorRef output) {
        for (int row = 0; row < input.shape().first(); row++) {
            int column = 0;
            int upperBound = F16_INT_SPECIES.loopBound(input.shape().last());
            for (; column < upperBound; column += F16_INT_SPECIES.length()) {
                IntVector bits = f16ToFloatBits(ShortVector.fromMemorySegment(BF16_SPECIES,
                        input.underlying().getMemorySegment(), memoryOffset(input, row, column), ByteOrder.LITTLE_ENDIAN));
                bits.reinterpretAsFloats().intoMemorySegment(output.underlying().getMemorySegment(),
                        memoryOffset(output, row, column), ByteOrder.LITTLE_ENDIAN);
            }
            for (; column < input.shape().last(); column++) {
                output.underlying().set(input.underlying().get(row, column), row, column);
            }
        }
    }

    private void reshapeF16ToBF16(TensorRef input, TensorRef output) {
        for (int row = 0; row < input.shape().first(); row++) {
            int column = 0;
            int upperBound = F16_INT_SPECIES.loopBound(input.shape().last());
            for (; column < upperBound; column += F16_INT_SPECIES.length()) {
                IntVector bits = f16ToFloatBits(ShortVector.fromMemorySegment(BF16_SPECIES,
                        input.underlying().getMemorySegment(), memoryOffset(input, row, column), ByteOrder.LITTLE_ENDIAN));
                IntVector rounded = roundFloatBitsToBf16(bits);
                ((ShortVector) rounded.lanewise(VectorOperators.LSHR, 16)
                        .convertShape(VectorOperators.I2S, BF16_SPECIES, 0))
                        .intoMemorySegment(output.underlying().getMemorySegment(), memoryOffset(output, row, column),
                                ByteOrder.LITTLE_ENDIAN);
            }
            for (; column < input.shape().last(); column++) {
                output.underlying().set(input.underlying().get(row, column), row, column);
            }
        }
    }

    private void reshapeF32ToF16(TensorRef input, TensorRef output) {
        for (int row = 0; row < input.shape().first(); row++) {
            int column = 0;
            int upperBound = F16_INT_SPECIES.loopBound(input.shape().last());
            for (; column < upperBound; column += F16_INT_SPECIES.length()) {
                FloatVector values = FloatVector.fromMemorySegment(F32_BF16_SPECIES,
                        input.underlying().getMemorySegment(), memoryOffset(input, row, column), ByteOrder.LITTLE_ENDIAN);
                floatBitsToF16(values.reinterpretAsInts()).intoMemorySegment(output.underlying().getMemorySegment(),
                        memoryOffset(output, row, column), ByteOrder.LITTLE_ENDIAN);
            }
            for (; column < input.shape().last(); column++) {
                output.underlying().set(input.underlying().get(row, column), row, column);
            }
        }
    }

    private void reshapeBF16ToF16(TensorRef input, TensorRef output) {
        for (int row = 0; row < input.shape().first(); row++) {
            int column = 0;
            int upperBound = F16_INT_SPECIES.loopBound(input.shape().last());
            for (; column < upperBound; column += F16_INT_SPECIES.length()) {
                ShortVector values = ShortVector.fromMemorySegment(BF16_SPECIES,
                        input.underlying().getMemorySegment(), memoryOffset(input, row, column), ByteOrder.LITTLE_ENDIAN);
                IntVector bits = ((IntVector) values.convertShape(VectorOperators.S2I, F16_INT_SPECIES, 0))
                        .lanewise(VectorOperators.LSHL, 16);
                floatBitsToF16(bits).intoMemorySegment(output.underlying().getMemorySegment(),
                        memoryOffset(output, row, column), ByteOrder.LITTLE_ENDIAN);
            }
            for (; column < input.shape().last(); column++) {
                output.underlying().set(input.underlying().get(row, column), row, column);
            }
        }
    }

    private IntVector f16ToFloatBits(ShortVector values) {
        IntVector half = ((IntVector) values.convertShape(VectorOperators.S2I, F16_INT_SPECIES, 0))
                .lanewise(VectorOperators.AND, 0xffff);
        IntVector sign = half.lanewise(VectorOperators.AND, 0x8000).lanewise(VectorOperators.LSHL, 16);
        IntVector exponent = half.lanewise(VectorOperators.LSHR, 10).lanewise(VectorOperators.AND, 0x1f);
        IntVector mantissa = half.lanewise(VectorOperators.AND, 0x3ff);

        IntVector normal = exponent.add(112).lanewise(VectorOperators.LSHL, 23)
                .or(mantissa.lanewise(VectorOperators.LSHL, 13));

        IntVector subnormalMantissa = mantissa;
        IntVector subnormalShift = IntVector.zero(F16_INT_SPECIES);
        for (int i = 0; i < 10; i++) {
            VectorMask<Integer> needsShift = subnormalMantissa.compare(VectorOperators.LT, 0x400);
            subnormalMantissa = subnormalMantissa.lanewise(VectorOperators.LSHL, 1, needsShift);
            subnormalShift = subnormalShift.sub(1, needsShift);
        }
        IntVector subnormal = subnormalShift.add(103).lanewise(VectorOperators.LSHL, 23)
                .or(subnormalMantissa.lanewise(VectorOperators.AND, 0x3ff).lanewise(VectorOperators.LSHL, 13));
        IntVector special = IntVector.broadcast(F16_INT_SPECIES, 0x7f800000)
                .or(mantissa.lanewise(VectorOperators.LSHL, 13));

        VectorMask<Integer> isZero = exponent.compare(VectorOperators.EQ, 0).and(mantissa.compare(VectorOperators.EQ, 0));
        VectorMask<Integer> isSubnormal = exponent.compare(VectorOperators.EQ, 0).and(mantissa.compare(VectorOperators.NE, 0));
        VectorMask<Integer> isSpecial = exponent.compare(VectorOperators.EQ, 31);
        IntVector result = normal.blend(subnormal, isSubnormal).blend(special, isSpecial).blend(sign, isZero);
        return result.or(sign);
    }

    private IntVector roundFloatBitsToBf16(IntVector bits) {
        return bits.add(BF16_ROUND_BIAS)
                .add(bits.lanewise(VectorOperators.LSHR, 16).lanewise(VectorOperators.AND, BF16_ROUND_LSB_MASK));
    }

    private ShortVector floatBitsToF16(IntVector bits) {
        IntVector sign = bits.lanewise(VectorOperators.LSHR, 16).lanewise(VectorOperators.AND, 0x8000);
        IntVector absolute = bits.lanewise(VectorOperators.AND, 0x7fffffff);
        IntVector normal = bits.sub(0x38000000).add(0x00001000)
                .lanewise(VectorOperators.LSHR, 13).lanewise(VectorOperators.AND, 0x7fff);
        VectorMask<Integer> overflow = absolute.compare(VectorOperators.GE, 0x47800000);
        VectorMask<Integer> nan = absolute.compare(VectorOperators.GT, 0x7f800000);
        VectorMask<Integer> underflow = absolute.compare(VectorOperators.LT, 0x33000000);
        VectorMask<Integer> subnormal = absolute.compare(VectorOperators.LT, 0x38800000);
        IntVector subnormalMantissa = absolute.lanewise(VectorOperators.AND, 0x7fffff).or(0x800000);
        IntVector shift = IntVector.broadcast(F16_INT_SPECIES, 126)
                .sub(absolute.lanewise(VectorOperators.LSHR, 23));
        IntVector subnormalValue = subnormalMantissa.add(IntVector.broadcast(F16_INT_SPECIES, 1)
                        .lanewise(VectorOperators.LSHL, shift.sub(1)))
                .lanewise(VectorOperators.LSHR, shift);
        IntVector infinity = IntVector.broadcast(F16_INT_SPECIES, 0x7c00);
        IntVector nanValue = infinity.or(absolute.lanewise(VectorOperators.LSHR, 13)
                .lanewise(VectorOperators.AND, 0x3ff));
        IntVector result = normal.blend(subnormalValue, subnormal).blend(IntVector.zero(F16_INT_SPECIES), underflow)
                .blend(infinity, overflow).blend(nanValue, nan)
                .or(sign);
        return (ShortVector) result.convertShape(VectorOperators.I2S, BF16_SPECIES, 0);
    }

    private void reshapeToI8(TensorRef input, TensorRef output) {
        TensorRef scale = Q8Layout.scale(output);
        Preconditions.checkArgument(scale != null, "I8 output must have scale sidecar");
        if (input.dType() == DType.F32) {
            reshapeF32ToI8(input, output, scale);
            return;
        }
        reshapeDenseToI8Scalar(input, output, scale);
    }

    private void reshapeF32ToI8(TensorRef input, TensorRef output, TensorRef scale) {
        Tensor inputTensor = input.underlying();
        MemorySegment inputSegment = inputTensor.getMemorySegment();
        MemorySegment outputSegment = output.underlying().getMemorySegment();
        for (int row = 0; row < input.shape().first(); row++) {
            for (int column = 0; column < input.shape().last(); column += Q8Layout.BLOCK_SIZE) {
                FloatVector v0 = FloatVector.fromMemorySegment(Q8_FLOAT_SPECIES, inputSegment,
                        memoryOffset(input, row, column), ByteOrder.LITTLE_ENDIAN);
                FloatVector v1 = FloatVector.fromMemorySegment(Q8_FLOAT_SPECIES, inputSegment,
                        memoryOffset(input, row, column + Q8Layout.BLOCK_SIZE / 2), ByteOrder.LITTLE_ENDIAN);
                float max = v0.abs().max(v1.abs()).reduceLanes(VectorOperators.MAX);
                float inverse = 127f / max;
                float factor = inverse != 0.0f ? 1.0f / inverse : 0.0f;
                scale.underlying().set(factor, row, Q8Layout.scaleColumn(column));

                FloatVector q0 = v0.mul(FloatVector.broadcast(Q8_FLOAT_SPECIES, inverse));
                FloatVector q1 = v1.mul(FloatVector.broadcast(Q8_FLOAT_SPECIES, inverse));
                long out0 = memoryOffset(output, row, column);
                long out1 = memoryOffset(output, row, column + Q8Layout.BLOCK_SIZE / 2);
                for (int lane = 0; lane < Q8_FLOAT_SPECIES.length(); lane++) {
                    outputSegment.set(ValueLayout.JAVA_BYTE, out0 + lane, (byte) Math.round(q0.lane(lane)));
                    outputSegment.set(ValueLayout.JAVA_BYTE, out1 + lane, (byte) Math.round(q1.lane(lane)));
                }
            }
        }
    }

    private void reshapeDenseToI8Scalar(TensorRef input, TensorRef output, TensorRef scale) {
        for (int row = 0; row < input.shape().first(); row++) {
            for (int column = 0; column < input.shape().last(); column += Q8Layout.BLOCK_SIZE) {
                float max = Float.MIN_VALUE;
                for (int i = 0; i < Q8Layout.BLOCK_SIZE; i++) {
                    float value = input.underlying().get(row, column + i);
                    float abs = value < 0 ? -value : value;
                    if (abs > max) {
                        max = abs;
                    }
                }
                float inverse = 127f / max;
                float factor = inverse != 0.0f ? 1.0f / inverse : 0.0f;
                scale.underlying().set(factor, row, Q8Layout.scaleColumn(column));
                for (int i = 0; i < Q8Layout.BLOCK_SIZE; i++) {
                    output.set(input.underlying().get(row, column + i), row, column + i);
                }
            }
        }
    }

    private void reshapeToQ4(TensorRef input, TensorRef output) {
        TensorRef scale = Q4Layout.scale(output);
        Preconditions.checkArgument(scale != null, "Q4 output must have scale sidecar");
        Q4Tensor outputTensor = (Q4Tensor) output.underlying();
        for (int row = 0; row < input.shape().first(); row++) {
            for (int column = 0; column < input.shape().last(); column += Q4Layout.BLOCK_SIZE) {
                float max = Float.MIN_VALUE;
                float amax = Float.MIN_VALUE;
                for (int i = 0; i < Q4Layout.BLOCK_SIZE; i++) {
                    float value = input.underlying().get(row, column + i);
                    float abs = value < 0 ? -value : value;
                    if (abs > amax) {
                        max = value;
                        amax = abs;
                    }
                }
                float factor = max / -8.0f;
                float inverse = factor != 0.0f ? 1.0f / factor : 0.0f;
                scale.underlying().set(factor, row, Q4Layout.blockIndex(column));
                for (int i = 0; i < Q4Layout.HALF_BLOCK; i++) {
                    int low = q4Nibble(input.underlying().get(row, column + i) * inverse);
                    int high = q4Nibble(input.underlying().get(row, column + Q4Layout.HALF_BLOCK + i) * inverse);
                    outputTensor.setPackedByte((byte) (low | (high << 4)), row, Q4Layout.blockIndex(column), i);
                }
            }
        }
    }

    private void reshapeQuantizedToDense(TensorRef input, TensorRef output) {
        if (output.dType() == DType.F32) {
            for (int row = 0; row < input.shape().first(); row++) {
                for (int column = 0; column < input.shape().last(); column++) {
                    output.underlying().set(input.underlying().get(row, column), row, column);
                }
            }
            return;
        }
        TensorRef f32 = new Lighter().allocate(DType.F32, input.shape());
        try {
            reshapeQuantizedToDense(input, f32);
            reshapeF32ToBF16(f32, output);
        } finally {
            f32.close();
        }
    }

    private int q4Nibble(float value) {
        return Math.min(15, (byte) (value + 8.5f));
    }

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
        if (result.dType() != DType.F32) {
            return Either.Left(OpSupport.Unsupported);
        }
        return switch (a.dType()) {
            case F32 -> switch (b.dType()) {
                case F32 -> {
                    new F32BatchDotProductGemmer(operation).matmul();
                    yield Either.Right(null);
                }
                case BF16 -> {
                    new F32BF16BatchDotProductGemmer(operation).matmul();
                    yield Either.Right(null);
                }
                case I8 -> {
                    TensorRef q8Scale = Q8Layout.scale(b);
                    if (q8Scale == null || !q8Aligned(operation)) {
                        yield Either.Left(OpSupport.Unsupported);
                    }
                    new F32Q8BatchDotProductGemmer(operation, q8Scale).matmul();
                    yield Either.Right(null);
                }
                case Q4 -> {
                    TensorRef q4Scale = Q4Layout.scale(b);
                    if (q4Scale == null || !q4Aligned(operation)) {
                        yield Either.Left(OpSupport.Unsupported);
                    }
                    new F32Q4BatchDotProductGemmer(operation, q4Scale).matmul();
                    yield Either.Right(null);
                }
                default -> Either.Left(OpSupport.Unsupported);
            };
            case BF16 -> switch (b.dType()) {
                case BF16 -> {
                    new BF16BF16BatchDotProductGemmer(operation).matmul();
                    yield Either.Right(null);
                }
                case Q4 -> {
                    TensorRef q4Scale = Q4Layout.scale(b);
                    if (q4Scale == null || !q4Aligned(operation)) {
                        yield Either.Left(OpSupport.Unsupported);
                    }
                    new BF16Q4BatchDotProductGemmer(operation, q4Scale).matmul();
                    yield Either.Right(null);
                }
                default -> Either.Left(OpSupport.Unsupported);
            };
            case I8 -> switch (b.dType()) {
                case Q4 -> {
                    TensorRef q8Scale = Q8Layout.scale(a);
                    TensorRef q4Scale = Q4Layout.scale(b);
                    if (q8Scale == null || q4Scale == null || !q4Aligned(operation)) {
                        yield Either.Left(OpSupport.Unsupported);
                    }
                    new I8Q4BatchDotProductGemmer(operation, q8Scale, q4Scale).matmul();
                    yield Either.Right(null);
                }
                default -> Either.Left(OpSupport.Unsupported);
            };
            default -> Either.Left(OpSupport.Unsupported);
        };
    }

    @Override
    public Either<OpSupport, Void> dotProductRows(TensorRef output, TensorRef input, TensorRef weights,
            int inputStart, int inputLength, int weightRowStart, int weightRowCount, int outputColumnStart) {
        return dotProductRows(output, input, weights, inputStart, inputStart, inputLength, weightRowStart,
                weightRowCount, outputColumnStart);
    }

    @Override
    public Either<OpSupport, Void> dotProductRows(TensorRef output, TensorRef input, TensorRef weights,
            int inputColumnStart, int weightColumnStart, int columnLength, int weightRowStart,
            int weightRowCount, int outputColumnStart) {
        if (output.dType() == DType.F32 && input.dType() == DType.I8 && weights.dType() == DType.Q4) {
            TensorRef inputScale = Q8Layout.scale(input);
            TensorRef weightScale = Q4Layout.scale(weights);
            if (inputScale == null || weightScale == null
                    || inputColumnStart % Q8Layout.BLOCK_SIZE != 0
                    || weightColumnStart % Q4Layout.BLOCK_SIZE != 0
                    || columnLength % Q8Layout.BLOCK_SIZE != 0) {
                return Either.Left(OpSupport.Unsupported);
            }
            new I8Q4DotProductRowsGemmer(output, input, inputScale, weights, weightScale,
                    inputColumnStart, weightColumnStart, columnLength, weightRowStart, weightRowCount,
                    outputColumnStart).matmul();
            return Either.Right(null);
        }
        if (output.dType() == DType.F32 && input.dType() == DType.F32 && weights.dType() == DType.Q4) {
            TensorRef weightScale = Q4Layout.scale(weights);
            if (weightScale == null
                    || weightColumnStart % Q4Layout.BLOCK_SIZE != 0
                    || columnLength % Q4Layout.BLOCK_SIZE != 0) {
                return Either.Left(OpSupport.Unsupported);
            }
            new F32Q4DotProductRowsGemmer(output, input, weights, weightScale,
                    inputColumnStart, weightColumnStart, columnLength, weightRowStart, weightRowCount,
                    outputColumnStart).matmul();
            return Either.Right(null);
        }
        if (output.dType() != DType.F32 || input.dType() != DType.F32
                || (weights.dType() != DType.F32 && weights.dType() != DType.I8)) {
            return Either.Left(OpSupport.Unsupported);
        }
        return batchDotProduct(new BatchDotProduct()
                .result(output)
                .a(input)
                .b(weights)
                .aColumnOffset(inputColumnStart)
                .bColumnOffset(weightColumnStart)
                .columnLength(columnLength)
                .resultRowOffset(outputColumnStart - weightRowStart)
                .bRowOffset(weightRowStart)
                .rowChunkSize(weightRowCount));
    }

    @Override
    public Either<OpSupport, Void> saxpy(float alpha, TensorRef x, TensorRef y, int xOffset, int yOffset,
            int length) {
        if (x.dType() != DType.F32 || y.dType() != DType.F32 || y.shape().first() != 1) {
            return Either.Left(OpSupport.Unsupported);
        }
        saxpyF32(alpha, x, 0, y, 0, xOffset, yOffset, length);
        return Either.Right(null);
    }

    @Override
    public Either<OpSupport, Void> saxpy(TensorRef alpha, TensorRef x, TensorRef y, int xOffset, int yOffset,
            int length, int alphaOffset, int xRowOffset, int batchSize) {
        if (alpha.dType() != DType.F32 || x.dType() != DType.F32 || y.dType() != DType.F32
                || y.shape().first() != 1) {
            return Either.Left(OpSupport.Unsupported);
        }
        for (int row = 0; row < batchSize; row++) {
            saxpyF32(alpha.get(0, alphaOffset + row), x, xRowOffset + row, y, 0,
                    xOffset, yOffset, length);
        }
        return Either.Right(null);
    }

    private static void saxpyF32(float alpha, TensorRef x, int xRow, TensorRef y, int yRow,
            int xOffset, int yOffset, int length) {
        FloatVector factor = FloatVector.broadcast(F32_SPECIES, alpha);
        MemorySegment xSegment = x.memorySegment();
        MemorySegment ySegment = y.memorySegment();
        long xBase = x.memorySegmentOffset(x.shape().getOffset(xRow, xOffset));
        long yBase = y.memorySegmentOffset(y.shape().getOffset(yRow, yOffset));
        int upper = F32_SPECIES.loopBound(length);
        int column = 0;
        for (; column < upper; column += F32_SPECIES.length()) {
            long xAddress = xBase + (long) column * Float.BYTES;
            long yAddress = yBase + (long) column * Float.BYTES;
            FloatVector values = FloatVector.fromMemorySegment(F32_SPECIES, xSegment, xAddress,
                    ByteOrder.LITTLE_ENDIAN);
            FloatVector target = FloatVector.fromMemorySegment(F32_SPECIES, ySegment, yAddress,
                    ByteOrder.LITTLE_ENDIAN);
            target.add(values.mul(factor)).intoMemorySegment(ySegment, yAddress, ByteOrder.LITTLE_ENDIAN);
        }
        if (column < length) {
            VectorMask<Float> mask = F32_SPECIES.indexInRange(column, length);
            long xAddress = xBase + (long) column * Float.BYTES;
            long yAddress = yBase + (long) column * Float.BYTES;
            FloatVector values = FloatVector.fromMemorySegment(F32_SPECIES, xSegment, xAddress,
                    ByteOrder.LITTLE_ENDIAN, mask);
            FloatVector target = FloatVector.fromMemorySegment(F32_SPECIES, ySegment, yAddress,
                    ByteOrder.LITTLE_ENDIAN, mask);
            target.add(values.mul(factor)).intoMemorySegment(ySegment, yAddress, ByteOrder.LITTLE_ENDIAN, mask);
        }
    }

    private static boolean q8Aligned(BatchDotProduct operation) {
        return operation.aColumnOffset() % Q8Layout.BLOCK_SIZE == 0
                && operation.bColumnOffset() % Q8Layout.BLOCK_SIZE == 0
                && operation.columnLength() % Q8Layout.BLOCK_SIZE == 0;
    }

    private static boolean q4Aligned(BatchDotProduct operation) {
        return operation.aColumnOffset() % Q4Layout.BLOCK_SIZE == 0
                && operation.bColumnOffset() % Q4Layout.BLOCK_SIZE == 0
                && operation.columnLength() % Q4Layout.BLOCK_SIZE == 0;
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

    private abstract static class BatchGemmer {
        final BatchDotProduct operation;

        BatchGemmer(BatchDotProduct operation) {
            this.operation = operation;
        }

        final void matmul() {
            TensorRef result = operation.result();
            int bEnd = operation.bRowOffset() + operation.rowChunkSize();
            for (int resultRow = 0; resultRow < result.shape().first(); resultRow++) {
                int aRow = operation.aRowOffset() + resultRow;
                for (int bRow = operation.bRowOffset(); bRow < bEnd; bRow++) {
                    result.set(dot(aRow, bRow), resultRow, bRow + operation.resultRowOffset());
                }
            }
        }

        abstract float dot(int aRow, int bRow);
    }

    private static final class F32BF16BatchDotProductGemmer extends BatchGemmer {
        F32BF16BatchDotProductGemmer(BatchDotProduct operation) {
            super(operation);
        }

        @Override
        float dot(int aRow, int bRow) {
            TensorRef a = operation.a();
            TensorRef b = operation.b();
            FloatVector acc = FloatVector.zero(F32_BF16_SPECIES);
            int k = 0;
            int upperBound = F32_BF16_SPECIES.loopBound(operation.columnLength());
            for (; k < upperBound; k += F32_BF16_SPECIES.length()) {
                FloatVector av = FloatVector.fromMemorySegment(F32_BF16_SPECIES, a.memorySegment(),
                        memoryOffset(a, aRow, operation.aColumnOffset() + k), ByteOrder.LITTLE_ENDIAN);
                FloatVector bv = bf16Vector(b, bRow, operation.bColumnOffset() + k);
                acc = av.fma(bv, acc);
            }
            float sum = acc.reduceLanes(VectorOperators.ADD);
            for (; k < operation.columnLength(); k++) {
                sum += a.get(aRow, operation.aColumnOffset() + k) * b.get(bRow, operation.bColumnOffset() + k);
            }
            return sum;
        }
    }

    private static final class BF16BF16BatchDotProductGemmer extends BatchGemmer {
        BF16BF16BatchDotProductGemmer(BatchDotProduct operation) {
            super(operation);
        }

        @Override
        float dot(int aRow, int bRow) {
            TensorRef a = operation.a();
            TensorRef b = operation.b();
            FloatVector acc = FloatVector.zero(F32_BF16_SPECIES);
            int k = 0;
            int upperBound = F32_BF16_SPECIES.loopBound(operation.columnLength());
            for (; k < upperBound; k += F32_BF16_SPECIES.length()) {
                FloatVector av = bf16Vector(a, aRow, operation.aColumnOffset() + k);
                FloatVector bv = bf16Vector(b, bRow, operation.bColumnOffset() + k);
                acc = av.fma(bv, acc);
            }
            float sum = acc.reduceLanes(VectorOperators.ADD);
            for (; k < operation.columnLength(); k++) {
                sum += a.get(aRow, operation.aColumnOffset() + k) * b.get(bRow, operation.bColumnOffset() + k);
            }
            return sum;
        }
    }

    private static final class F32Q4BatchDotProductGemmer extends BatchGemmer {
        private final TensorRef weightScale;

        F32Q4BatchDotProductGemmer(BatchDotProduct operation, TensorRef weightScale) {
            super(operation);
            this.weightScale = weightScale;
        }

        @Override
        float dot(int aRow, int bRow) {
            TensorRef a = operation.a();
            TensorRef b = operation.b();
            FloatVector sum = FloatVector.zero(FloatVector.SPECIES_512);
            int aColumn = operation.aColumnOffset();
            int bColumn = operation.bColumnOffset();
            int end = operation.aColumnOffset() + operation.columnLength();
            for (; aColumn < end; aColumn += Q4Layout.BLOCK_SIZE, bColumn += Q4Layout.BLOCK_SIZE) {
                float scale = weightScale.get(bRow, Q4Layout.blockIndex(bColumn));
                ByteVector packedWeights = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, b.memorySegment(),
                        memoryOffset(b, bRow, bColumn), ByteOrder.LITTLE_ENDIAN);
                FloatVector low = (FloatVector) packedWeights.lanewise(VectorOperators.AND, Q4_MASK)
                        .sub(Q4_ZERO).convertShape(VectorOperators.B2F, FloatVector.SPECIES_512, 0);
                FloatVector high = (FloatVector) packedWeights.lanewise(VectorOperators.ASHR, Q4_SHIFT)
                        .lanewise(VectorOperators.AND, Q4_MASK).sub(Q4_ZERO)
                        .convertShape(VectorOperators.B2F, FloatVector.SPECIES_512, 0);
                FloatVector inputLow = FloatVector.fromMemorySegment(FloatVector.SPECIES_512, a.memorySegment(),
                        memoryOffset(a, aRow, aColumn), ByteOrder.LITTLE_ENDIAN);
                FloatVector inputHigh = FloatVector.fromMemorySegment(FloatVector.SPECIES_512, a.memorySegment(),
                        memoryOffset(a, aRow, aColumn + Q4Layout.HALF_BLOCK), ByteOrder.LITTLE_ENDIAN);
                sum = FloatVector.broadcast(FloatVector.SPECIES_512, scale)
                        .fma(low.mul(inputLow).add(high.mul(inputHigh)), sum);
            }
            return sum.reduceLanes(VectorOperators.ADD);
        }
    }

    private static final class BF16Q4BatchDotProductGemmer extends BatchGemmer {
        private final TensorRef weightScale;

        BF16Q4BatchDotProductGemmer(BatchDotProduct operation, TensorRef weightScale) {
            super(operation);
            this.weightScale = weightScale;
        }

        @Override
        float dot(int aRow, int bRow) {
            TensorRef a = operation.a();
            TensorRef b = operation.b();
            FloatVector sum = FloatVector.zero(FloatVector.SPECIES_512);
            int aColumn = operation.aColumnOffset();
            int bColumn = operation.bColumnOffset();
            int end = operation.aColumnOffset() + operation.columnLength();
            for (; aColumn < end; aColumn += Q4Layout.BLOCK_SIZE, bColumn += Q4Layout.BLOCK_SIZE) {
                float scale = weightScale.get(bRow, Q4Layout.blockIndex(bColumn));
                ByteVector packedWeights = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, b.memorySegment(),
                        memoryOffset(b, bRow, bColumn), ByteOrder.LITTLE_ENDIAN);
                FloatVector low = (FloatVector) packedWeights.lanewise(VectorOperators.AND, Q4_MASK)
                        .sub(Q4_ZERO).convertShape(VectorOperators.B2F, FloatVector.SPECIES_512, 0);
                FloatVector high = (FloatVector) packedWeights.lanewise(VectorOperators.ASHR, Q4_SHIFT)
                        .lanewise(VectorOperators.AND, Q4_MASK).sub(Q4_ZERO)
                        .convertShape(VectorOperators.B2F, FloatVector.SPECIES_512, 0);
                FloatVector inputLow = bf16Vector512(a, aRow, aColumn);
                FloatVector inputHigh = bf16Vector512(a, aRow, aColumn + Q4Layout.HALF_BLOCK);
                sum = FloatVector.broadcast(FloatVector.SPECIES_512, scale)
                        .fma(low.mul(inputLow).add(high.mul(inputHigh)), sum);
            }
            return sum.reduceLanes(VectorOperators.ADD);
        }
    }

    private static final class I8Q4BatchDotProductGemmer extends BatchGemmer {
        private final TensorRef inputScale;
        private final TensorRef weightScale;

        I8Q4BatchDotProductGemmer(BatchDotProduct operation, TensorRef inputScale, TensorRef weightScale) {
            super(operation);
            this.inputScale = inputScale;
            this.weightScale = weightScale;
        }

        @Override
        float dot(int aRow, int bRow) {
            TensorRef a = operation.a();
            TensorRef b = operation.b();
            FloatVector sum = FloatVector.zero(FloatVector.SPECIES_512);
            int aColumn = operation.aColumnOffset();
            int bColumn = operation.bColumnOffset();
            int end = operation.aColumnOffset() + operation.columnLength();
            for (; aColumn < end; aColumn += Q8Layout.BLOCK_SIZE, bColumn += Q4Layout.BLOCK_SIZE) {
                float scale = inputScale.get(aRow, Q8Layout.scaleColumn(aColumn))
                        * weightScale.get(bRow, Q4Layout.blockIndex(bColumn));
                ByteVector inputLowBytes = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, a.memorySegment(),
                        memoryOffset(a, aRow, aColumn), ByteOrder.LITTLE_ENDIAN);
                ByteVector inputHighBytes = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, a.memorySegment(),
                        memoryOffset(a, aRow, aColumn + Q4Layout.HALF_BLOCK), ByteOrder.LITTLE_ENDIAN);
                ShortVector inputLow = (ShortVector) inputLowBytes
                        .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0);
                ShortVector inputHigh = (ShortVector) inputHighBytes
                        .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0);
                ByteVector packedWeights = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, b.memorySegment(),
                        memoryOffset(b, bRow, bColumn), ByteOrder.LITTLE_ENDIAN);
                ShortVector low = (ShortVector) packedWeights.lanewise(VectorOperators.AND, Q4_MASK)
                        .sub(Q4_ZERO).convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0);
                ShortVector high = (ShortVector) packedWeights.lanewise(VectorOperators.ASHR, Q4_SHIFT)
                        .lanewise(VectorOperators.AND, Q4_MASK).sub(Q4_ZERO)
                        .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0);
                ShortVector products = low.mul(inputLow).add(high.mul(inputHigh));
                FloatVector block = (FloatVector) products.convertShape(VectorOperators.S2F, FloatVector.SPECIES_512, 0);
                sum = FloatVector.broadcast(FloatVector.SPECIES_512, scale).fma(block, sum);
            }
            return sum.reduceLanes(VectorOperators.ADD);
        }
    }

    private static FloatVector bf16Vector(TensorRef tensor, int row, int column) {
        return ShortVector.fromMemorySegment(BF16_SPECIES, tensor.memorySegment(), memoryOffset(tensor, row, column),
                        ByteOrder.LITTLE_ENDIAN)
                .convertShape(VectorOperators.S2I, IntVector.SPECIES_256, 0)
                .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_256)
                .reinterpretAsFloats();
    }

    private static FloatVector bf16Vector512(TensorRef tensor, int row, int column) {
        return ShortVector.fromMemorySegment(ShortVector.SPECIES_256, tensor.memorySegment(),
                        memoryOffset(tensor, row, column), ByteOrder.LITTLE_ENDIAN)
                .convertShape(VectorOperators.S2I, IntVector.SPECIES_512, 0)
                .lanewise(VectorOperators.LSHL, 16)
                .reinterpretAsFloats();
    }

    private record I8Q4DotProductRowsGemmer(TensorRef output, TensorRef input, TensorRef inputScale,
            TensorRef weights, TensorRef weightScale, int inputColumnStart, int weightColumnStart,
            int columnLength, int weightRowStart, int weightRowCount, int outputColumnStart) {
        private void matmul() {
            for (int inputRow = 0; inputRow < input.shape().first(); inputRow++) {
                for (int row = 0; row < weightRowCount; row++) {
                    int weightRow = weightRowStart + row;
                    output.set(dot(inputRow, weightRow), inputRow, outputColumnStart + row);
                }
            }
        }

        private float dot(int inputRow, int weightRow) {
            FloatVector sum = FloatVector.zero(FloatVector.SPECIES_512);
            int inputColumn = inputColumnStart;
            int weightColumn = weightColumnStart;
            int end = inputColumnStart + columnLength;
            for (; inputColumn < end;
                    inputColumn += Q8Layout.BLOCK_SIZE, weightColumn += Q4Layout.BLOCK_SIZE) {
                float scale = inputScale.get(inputRow, Q8Layout.scaleColumn(inputColumn))
                        * weightScale.get(weightRow, Q4Layout.blockIndex(weightColumn));
                ByteVector inputLowBytes = ByteVector.fromMemorySegment(ByteVector.SPECIES_128,
                        input.memorySegment(), memoryOffset(input, inputRow, inputColumn), ByteOrder.LITTLE_ENDIAN);
                ByteVector inputHighBytes = ByteVector.fromMemorySegment(ByteVector.SPECIES_128,
                        input.memorySegment(), memoryOffset(input, inputRow,
                                inputColumn + Q4Layout.HALF_BLOCK), ByteOrder.LITTLE_ENDIAN);
                ShortVector inputLow = (ShortVector) inputLowBytes
                        .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0);
                ShortVector inputHigh = (ShortVector) inputHighBytes
                        .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0);
                ByteVector packedWeights = ByteVector.fromMemorySegment(ByteVector.SPECIES_128,
                        weights.memorySegment(), memoryOffset(weights, weightRow, weightColumn),
                        ByteOrder.LITTLE_ENDIAN);
                ShortVector low = (ShortVector) packedWeights.lanewise(VectorOperators.AND, Q4_MASK)
                        .sub(Q4_ZERO).convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0);
                ShortVector high = (ShortVector) packedWeights.lanewise(VectorOperators.ASHR, Q4_SHIFT)
                        .lanewise(VectorOperators.AND, Q4_MASK).sub(Q4_ZERO)
                        .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0);
                ShortVector products = low.mul(inputLow).add(high.mul(inputHigh));
                FloatVector block = (FloatVector) products.convertShape(VectorOperators.S2F,
                        FloatVector.SPECIES_512, 0);
                sum = FloatVector.broadcast(FloatVector.SPECIES_512, scale).fma(block, sum);
            }
            return sum.reduceLanes(VectorOperators.ADD);
        }
    }

    private record F32Q4DotProductRowsGemmer(TensorRef output, TensorRef input, TensorRef weights,
            TensorRef weightScale, int inputColumnStart, int weightColumnStart, int columnLength,
            int weightRowStart, int weightRowCount, int outputColumnStart) {
        private void matmul() {
            for (int inputRow = 0; inputRow < input.shape().first(); inputRow++) {
                for (int row = 0; row < weightRowCount; row++) {
                    int weightRow = weightRowStart + row;
                    output.set(dot(inputRow, weightRow), inputRow, outputColumnStart + row);
                }
            }
        }

        private float dot(int inputRow, int weightRow) {
            FloatVector sum = FloatVector.zero(FloatVector.SPECIES_512);
            int inputColumn = inputColumnStart;
            int weightColumn = weightColumnStart;
            int end = inputColumnStart + columnLength;
            for (; inputColumn < end;
                    inputColumn += Q4Layout.BLOCK_SIZE, weightColumn += Q4Layout.BLOCK_SIZE) {
                float scale = weightScale.get(weightRow, Q4Layout.blockIndex(weightColumn));
                ByteVector packedWeights = ByteVector.fromMemorySegment(ByteVector.SPECIES_128,
                        weights.memorySegment(), memoryOffset(weights, weightRow, weightColumn),
                        ByteOrder.LITTLE_ENDIAN);
                FloatVector low = (FloatVector) packedWeights.lanewise(VectorOperators.AND, Q4_MASK)
                        .sub(Q4_ZERO).convertShape(VectorOperators.B2F, FloatVector.SPECIES_512, 0);
                FloatVector high = (FloatVector) packedWeights.lanewise(VectorOperators.ASHR, Q4_SHIFT)
                        .lanewise(VectorOperators.AND, Q4_MASK).sub(Q4_ZERO)
                        .convertShape(VectorOperators.B2F, FloatVector.SPECIES_512, 0);
                FloatVector inputLow = FloatVector.fromMemorySegment(FloatVector.SPECIES_512,
                        input.memorySegment(), memoryOffset(input, inputRow, inputColumn), ByteOrder.LITTLE_ENDIAN);
                FloatVector inputHigh = FloatVector.fromMemorySegment(FloatVector.SPECIES_512,
                        input.memorySegment(), memoryOffset(input, inputRow, inputColumn + Q4Layout.HALF_BLOCK),
                        ByteOrder.LITTLE_ENDIAN);
                FloatVector block = low.mul(inputLow).add(high.mul(inputHigh));
                sum = FloatVector.broadcast(FloatVector.SPECIES_512, scale).fma(block, sum);
            }
            return sum.reduceLanes(VectorOperators.ADD);
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
        if (target.dType() == DType.F16) {
            scaleF16(factor, target, offset, length);
            return Either.Right(null);
        }
        return Either.Left(OpSupport.Unsupported);
    }

    private void scaleF16(float factor, TensorRef target, int offset, int length) {
        FloatVector scale = FloatVector.broadcast(F32_BF16_SPECIES, factor);
        int end = offset + length;
        int column = offset;
        int upperBound = offset + F16_INT_SPECIES.loopBound(length);
        for (int row = 0; row < target.shape().first(); row++) {
            column = offset;
            for (; column < upperBound; column += F16_INT_SPECIES.length()) {
                IntVector values = f16ToFloatBits(ShortVector.fromMemorySegment(BF16_SPECIES,
                        target.underlying().getMemorySegment(), memoryOffset(target, row, column), ByteOrder.LITTLE_ENDIAN));
                FloatVector scaled = values.reinterpretAsFloats().mul(scale);
                floatBitsToF16(scaled.reinterpretAsInts()).intoMemorySegment(target.underlying().getMemorySegment(),
                        memoryOffset(target, row, column), ByteOrder.LITTLE_ENDIAN);
            }
            for (; column < end; column++) {
                target.underlying().set(target.underlying().get(row, column) * factor, row, column);
            }
        }
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
        return tensor.memorySegmentOffset(tensor.shape().getOffset(row, column));
    }
}
