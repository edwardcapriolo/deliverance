package io.teknek.deliverance.tensor2;

import io.teknek.dysfx.Either;

public interface TensorOps {
     default Either<OpSupport, Void> sum(TensorRef input, int row, int offset, int length, TensorRef output) {
          return Either.Left(OpSupport.Unsupported);
     }

     default Either<OpSupport, Void> exp(TensorRef input, TensorRef output, int offset, int length) {
          return Either.Left(OpSupport.Unsupported);
     }

     default Either<OpSupport, Void> max(TensorRef input, int row, int offset, int length, TensorRef output) {
          return Either.Left(OpSupport.Unsupported);
     }

     default Either<OpSupport, Void> argMax(TensorRef input, TensorRef output, int offset, int length) {
          return Either.Left(OpSupport.Unsupported);
     }

     default Either<OpSupport, Void> accumulate(TensorRef a, TensorRef b, int offset, int length) {
          return Either.Left(OpSupport.Unsupported);
     }

     Either<OpSupport, Void> multiplyAccumulate(TensorRef a, TensorRef b, int offset, int length);

     default Either<OpSupport, Void> reshape(TensorRef input, TensorRef output) {
          return Either.Left(OpSupport.Unsupported);
     }

     default Either<OpSupport, Void> batchDotProduct(BatchDotProduct operation) {
          return Either.Left(OpSupport.Unsupported);
     }

     default Either<OpSupport, Void> dotProductRows(TensorRef output, TensorRef input, TensorRef weights,
             int inputStart, int inputLength, int weightRowStart, int weightRowCount, int outputColumnStart) {
          return dotProductRows(output, input, weights, inputStart, inputStart, inputLength, weightRowStart,
                  weightRowCount, outputColumnStart);
     }

     default Either<OpSupport, Void> dotProductRows(TensorRef output, TensorRef input, TensorRef weights,
             int inputColumnStart, int weightColumnStart, int columnLength, int weightRowStart,
             int weightRowCount, int outputColumnStart) {
           return Either.Left(OpSupport.Unsupported);
      }

      /** Provider-level paired projection; providers without a fused path remain unsupported. */
      default Either<OpSupport, Void> dotProductBatchChunk(DotProductBatchChunk operation) {
           return Either.Left(OpSupport.Unsupported);
      }

      default boolean supportsDotProductBatchChunk(DotProductBatchChunk operation) {
           return false;
      }


     Either<OpSupport, Void> scale(float factor, TensorRef target, int offset, int length);

     default Either<OpSupport, Void> saxpy(float alpha, TensorRef x, TensorRef y,
             int xOffset, int yOffset, int length) {
          return Either.Left(OpSupport.Unsupported);
     }

     default Either<OpSupport, Void> saxpy(TensorRef alpha, TensorRef x, TensorRef y,
             int xOffset, int yOffset, int length, int alphaOffset, int xRowOffset, int batchSize) {
          return Either.Left(OpSupport.Unsupported);
     }
}
