package io.teknek.deliverance.tensor2;

import com.google.common.base.Preconditions;
import io.teknek.deliverance.DType;
import io.teknek.dysfx.Either;

class NaiveOps implements TensorOps {

    @Override
    public Either<OpSupport, Void> multiplyAccumulate(TensorRef a, TensorRef b, int offset, int length) {
        Preconditions.checkArgument(a.dims() == b.dims());
        Preconditions.checkArgument(a.shape().last() == b.shape().last());
        Preconditions.checkArgument(b.shape().first() == 1 || a.shape().first() == b.shape().first());
        Preconditions.checkArgument(offset >= 0 && length >= 0 && offset + length <= a.shape().last());

        boolean isBatch = b.shape().first() > 1;
        for (int ai = 0; ai < a.shape().first(); ai++) {
            int bi = isBatch ? ai : 0;
            for (int i = offset; i < offset + length; ++i) {
                a.underlying().set(a.underlying().get(ai, i) * b.underlying().get(bi, i), ai, i);
            }
        }
        return Either.Right(null);
    }

    @Override
    public Either<OpSupport, Void> reshape(TensorRef input, TensorRef output) {
        Preconditions.checkArgument(input.dims() == 2 && output.dims() == 2, "Reshape requires 2D tensors");
        Preconditions.checkArgument(input.shape().equals(output.shape()), "Input and output shapes must match");
        if (output.dType() == DType.Q4) {
            reshapeQ4(input, output);
            return Either.Right(null);
        }
        if (output.dType() == DType.I8) {
            reshapeI8(input, output);
            return Either.Right(null);
        }
        if (output.dType() == DType.F16 || output.dType() == DType.F32 || output.dType() == DType.BF16) {
            for (int row = 0; row < input.shape().first(); row++) {
                for (int column = 0; column < input.shape().last(); column++) {
                    output.underlying().set(input.underlying().get(row, column), row, column);
                }
            }
            return Either.Right(null);
        }
        return Either.Left(OpSupport.Unsupported);
    }

    private void reshapeI8(TensorRef input, TensorRef output) {
        TensorRef scale = Q8Layout.scale(output);
        Preconditions.checkArgument(scale != null, "I8 output must have scale sidecar");
        I8Tensor outputTensor = (I8Tensor) output.underlying();
        for (int row = 0; row < input.shape().first(); row++) {
            for (int column = 0; column < input.shape().last(); column += Q8Layout.BLOCK_SIZE) {
                float max = 0.0f;
                for (int i = 0; i < Q8Layout.BLOCK_SIZE; i++) {
                    float value = input.underlying().get(row, column + i);
                    float abs = Math.abs(value);
                    if (abs > max) {
                        max = abs;
                    }
                }
                float factor = max / Byte.MAX_VALUE;
                float inverse = factor != 0.0f ? 1.0f / factor : 0.0f;
                scale.underlying().set(factor, row, Q8Layout.scaleColumn(column));
                for (int i = 0; i < Q8Layout.BLOCK_SIZE; i++) {
                    outputTensor.setRawByte((byte) Math.round(input.underlying().get(row, column + i) * inverse),
                            row, column + i);
                }
            }
        }
    }

    private void reshapeQ4(TensorRef input, TensorRef output) {
        TensorRef scale = Q4Layout.scale(output);
        Preconditions.checkArgument(scale != null, "Q4 output must have scale sidecar");
        Q4Tensor outputTensor = (Q4Tensor) output.underlying();
        for (int row = 0; row < input.shape().first(); row++) {
            for (int column = 0; column < input.shape().last(); column += Q4Layout.BLOCK_SIZE) {
                float max = 0.0f;
                for (int i = 0; i < Q4Layout.BLOCK_SIZE; i++) {
                    float value = input.underlying().get(row, column + i);
                    float abs = Math.abs(value);
                    if (abs > Math.abs(max)) {
                        max = value;
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

    private int q4Nibble(float value) {
        return Math.max(0, Math.min(15, (int) (value + 8.5f)));
    }

    @Override
    public Either<OpSupport, Void> batchDotProduct(BatchDotProduct operation) {
        TensorRef result = operation.result();
        TensorRef a = operation.a();
        TensorRef b = operation.b();
        if (result.dType() != DType.F32) {
            return Either.Left(OpSupport.Unsupported);
        }
        Preconditions.checkArgument(result.dims() == 2 && a.dims() == 2 && b.dims() == 2);
        int bEnd = operation.bRowOffset() + operation.rowChunkSize();
        for (int resultRow = 0; resultRow < result.shape().first(); resultRow++) {
            int aRow = operation.aRowOffset() + resultRow;
            for (int bRow = operation.bRowOffset(); bRow < bEnd; bRow++) {
                float sum = 0.0f;
                for (int k = 0; k < operation.columnLength(); k++) {
                    sum += a.underlying().get(aRow, operation.aColumnOffset() + k)
                            * b.underlying().get(bRow, operation.bColumnOffset() + k);
                }
                result.underlying().set(sum, resultRow, bRow + operation.resultRowOffset());
            }
        }
        return Either.Right(null);
    }

    @Override
    public Either<OpSupport, Void> scale(float factor, TensorRef target, int offset, int length) {
        Preconditions.checkArgument(offset >= 0 && length >= 0 && offset + length <= target.shape().last());
        for (int row = 0; row < target.shape().first(); row++) {
            for (int column = offset; column < offset + length; column++) {
                target.underlying().set(target.underlying().get(row, column) * factor, row, column);
            }
        }
        return Either.Right(null);
    }
}
