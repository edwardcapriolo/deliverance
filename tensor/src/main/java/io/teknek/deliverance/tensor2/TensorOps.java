package io.teknek.deliverance.tensor2;

import io.teknek.dysfx.Either;

public interface TensorOps {
     Either<OpSupport, Void> multiplyAccumulate(TensorRef a, TensorRef b, int offset, int length);

     default Either<OpSupport, Void> reshape(TensorRef input, TensorRef output) {
          return Either.Left(OpSupport.Unsupported);
     }

     default Either<OpSupport, Void> batchDotProduct(BatchDotProduct operation) {
          return Either.Left(OpSupport.Unsupported);
     }

     Either<OpSupport, Void> scale(float factor, TensorRef target, int offset, int length);
}
