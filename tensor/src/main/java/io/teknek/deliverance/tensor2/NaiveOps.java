package io.teknek.deliverance.tensor2;

import com.google.common.base.Preconditions;
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
}
