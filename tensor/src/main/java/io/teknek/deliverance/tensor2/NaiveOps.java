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
