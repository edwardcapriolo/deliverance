package io.teknek.deliverance.tensor2;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.operations.tensor2native.Tensor2Native;
import io.teknek.deliverance.tensor.operations.util.JarSupport;
import io.teknek.dysfx.Either;

import java.lang.foreign.MemorySegment;

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
        if (result.dType() != DType.F32 || a.dType() != DType.F32) {
            return Either.Left(OpSupport.Unsupported);
        }
        TensorRef q8Scale = Q8Layout.scale(b);
        if (b.dType() == DType.I8 && q8Scale != null && q8Aligned(operation)) {
            return batchDotProductF32Q8(operation, q8Scale);
        }
        if (b.dType() != DType.F32) {
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

    @Override
    public Either<OpSupport, Void> scale(float factor, TensorRef target, int offset, int length) {
        if (!loaded) {
            return Either.Left(OpSupport.Unsupported);
        }
        try {
            int status;
            if (target.dType() == DType.F32) {
                status = Tensor2Native.tensor2_scale_f32(target.underlying().getMemorySegment(), factor,
                        (int) target.shape().first(), offset, length, target.stride());
            } else if (target.dType() == DType.BF16) {
                status = Tensor2Native.tensor2_scale_bf16(target.underlying().getMemorySegment(), factor,
                        (int) target.shape().first(), offset, length, target.stride());
            } else {
                return Either.Left(OpSupport.Unsupported);
            }
            return status == Tensor2Native.TENSOR2_OK() ? Either.Right(null) : Either.Left(OpSupport.Unsupported);
        } catch (LinkageError | RuntimeException e) {
            return Either.Left(OpSupport.Unsupported);
        }
    }

    private Either<OpSupport, Void> batchDotProductF32Q8(BatchDotProduct operation, TensorRef q8Scale) {
        TensorRef result = operation.result();
        TensorRef a = operation.a();
        TensorRef b = operation.b();
        MemorySegment resultSegment = result.underlying().getMemorySegment()
                .asSlice(memoryOffset(result, 0, operation.resultRowOffset() + operation.bRowOffset()));
        MemorySegment aSegment = a.underlying().getMemorySegment()
                .asSlice(memoryOffset(a, operation.aRowOffset(), operation.aColumnOffset()));
        MemorySegment bSegment = b.underlying().getMemorySegment()
                .asSlice(memoryOffset(b, operation.bRowOffset(), operation.bColumnOffset()));
        MemorySegment bScales = q8Scale.underlying().getMemorySegment().asSlice(memoryOffset(q8Scale,
                operation.bRowOffset(), Q8Layout.scaleColumn(operation.bColumnOffset())));
        try {
            int status = Tensor2Native.tensor2_batch_dot_f32_q8(resultSegment, aSegment, bSegment, bScales,
                    (int) result.shape().first(), operation.rowChunkSize(), operation.columnLength(),
                    result.stride(), a.stride(), b.stride(), q8Scale.stride());
            return status == Tensor2Native.TENSOR2_OK() ? Either.Right(null) : Either.Left(OpSupport.Unsupported);
        } catch (LinkageError | RuntimeException e) {
            return Either.Left(OpSupport.Unsupported);
        }
    }

    private static boolean q8Aligned(BatchDotProduct operation) {
        return operation.aColumnOffset() % Q8Layout.BLOCK_SIZE == 0
                && operation.bColumnOffset() % Q8Layout.BLOCK_SIZE == 0
                && operation.columnLength() % Q8Layout.BLOCK_SIZE == 0;
    }

    private static long memoryOffset(TensorRef tensor, int row, int column) {
        return tensor.underlying().getMemorySegmentOffset(tensor.shape().getOffset(row, column));
    }
}
