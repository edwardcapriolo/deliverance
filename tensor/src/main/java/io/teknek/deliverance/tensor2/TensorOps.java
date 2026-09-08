package io.teknek.deliverance.tensor2;

import io.teknek.dysfx.Either;

public interface TensorOps {
     Either<OpSupport, Void> multiplyAccumulate(TensorRef a, TensorRef b, int offset, int length);
}
