package io.teknek.deliverance.tensor2;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.operations.tensor2native.Tensor2Native;
import io.teknek.deliverance.tensor.operations.util.JarSupport;
import io.teknek.dysfx.Either;

public class NativeOps implements TensorOps {
    private static final boolean loaded = loadOnce();

    private static boolean loadOnce() {
        boolean libraryLoaded = JarSupport.maybeLoadLibrary("deliverance_tensor2");
        if (!libraryLoaded) {
            try {
                System.loadLibrary("deliverance_tensor2");
                libraryLoaded = true;
            } catch (UnsatisfiedLinkError ignored) {
                return false;
            }
        }
        try {
            Tensor2Native.tensor2_batch_dot_f32_f32$address();
            return true;
        } catch (LinkageError | RuntimeException e) {
            return false;
        }
    }

    public NativeOps() {
        if (!loaded) {
            throw new IllegalStateException("tensor2 native operations are not available");
        }
    }

    public static boolean isAvailable() {
        return loaded;
    }

    @Override
    public Either<OpSupport, Void> multiplyAccumulate(TensorRef a, TensorRef b, int offset, int length) {
        return Either.Left(OpSupport.Unsupported);
    }

    @Override
    public Either<OpSupport, Void> batchDotProduct(BatchDotProduct operation) {
        if (!loaded) {
            return Either.Left(OpSupport.Unsupported);
        }
        TensorRef result = operation.result();
        TensorRef a = operation.a();
        TensorRef b = operation.b();
        if (result.dType() != DType.F32 || a.dType() != DType.F32 || b.dType() != DType.F32) {
            return Either.Left(OpSupport.Unsupported);
        }
        int status = Tensor2Native.tensor2_batch_dot_f32_f32(
                result.underlying().getMemorySegment(),
                a.underlying().getMemorySegment(),
                b.underlying().getMemorySegment(),
                (int) result.shape().first(),
                operation.aRowOffset(),
                operation.aColumnOffset(),
                operation.bColumnOffset(),
                operation.columnLength(),
                operation.resultRowOffset(),
                operation.bRowOffset(),
                operation.rowChunkSize(),
                result.stride(),
                a.stride(),
                b.stride());
        return status == Tensor2Native.TENSOR2_OK() ? Either.Right(null) : Either.Left(OpSupport.Unsupported);
    }
}
