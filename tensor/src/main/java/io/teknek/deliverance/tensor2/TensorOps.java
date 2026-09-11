package io.teknek.deliverance.tensor2;

import io.teknek.dysfx.Either;

public interface TensorOps {
     Either<OpSupport, Void> multiplyAccumulate(TensorRef a, TensorRef b, int offset, int length);

     default Either<OpSupport, Void> scale(float factor, TensorRef target, int offset, int length) {
          return Either.Left(OpSupport.Unsupported);
     }
}
