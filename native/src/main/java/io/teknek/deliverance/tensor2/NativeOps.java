package io.teknek.deliverance.tensor2;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.operations.tensor2native.Tensor2Native;
import io.teknek.deliverance.tensor.operations.cnative.NativeSimd;
import io.teknek.deliverance.tensor.operations.util.JarSupport;
import io.teknek.dysfx.Either;

import java.lang.foreign.MemorySegment;
public class NativeOps implements TensorOps, CompositeOpsProvider {
    private static Throwable loadFailure;
    private static final boolean loaded = loadOnce();

    private static boolean loadOnce() {
        boolean libraryLoaded = JarSupport.maybeLoadLibrary("deliverance_tensor2");
        if (!libraryLoaded) {
            try {
                System.loadLibrary("deliverance_tensor2");
                libraryLoaded = true;
            } catch (UnsatisfiedLinkError ignored) {
                loadFailure = ignored;
                return false;
            }
        }
        try {
            Tensor2Native.tensor2_batch_dot_f32_f32$address();
            return true;
        } catch (LinkageError | RuntimeException e) {
            loadFailure = e;
            return false;
        }
    }

    public NativeOps() {
        if (!loaded) {
            throw new IllegalStateException("tensor2 native operations are not available", loadFailure);
        }
    }

    @Override
    public Either<OpSupport, Void> exp(TensorRef input, TensorRef output, int offset, int length) {
        if (input.dType() != DType.F32 || output.dType() != DType.F32
                || !input.shape().equals(output.shape()) || input.dims() != 2) {
            return Either.Left(OpSupport.Unsupported);
        }
        int status = Tensor2Native.tensor2_exp_f32(baseSegment(input), baseSegment(output),
                (int) input.shape().first(), offset, length, input.stride(), output.stride());
        return status == Tensor2Native.TENSOR2_OK() ? Either.Right(null) : Either.Left(OpSupport.Unsupported);
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
        if (result.dType() == DType.F32 && a.dType() == DType.F32 && b.dType() == DType.Q4) {
            TensorRef scale = Q4Layout.scale(b);
            if (scale == null) {
                return Either.Left(OpSupport.Unsupported);
            }
            int status = Tensor2Native.tensor2_gemm_f32_q4(baseSegment(result), baseSegment(a), baseSegment(b),
                    baseSegment(scale), (int) result.shape().first(), operation.aRowOffset(),
                    operation.aColumnOffset(), operation.bColumnOffset(), operation.columnLength(),
                    operation.resultRowOffset(), operation.bRowOffset(), operation.rowChunkSize(), result.stride(),
                    a.stride(), b.stride(), scale.stride());
            return status == Tensor2Native.TENSOR2_OK() ? Either.Right(null) : Either.Left(OpSupport.Unsupported);
        }
        if (result.dType() == DType.F32 && a.dType() == DType.I8 && b.dType() == DType.Q4) {
            TensorRef inputScale = Q8Layout.scale(a);
            TensorRef weightScale = Q4Layout.scale(b);
            if (inputScale == null || weightScale == null) {
                return Either.Left(OpSupport.Unsupported);
            }
            int status = Tensor2Native.tensor2_gemm_i8_q4(baseSegment(result), baseSegment(a),
                    baseSegment(inputScale), baseSegment(b), baseSegment(weightScale), (int) result.shape().first(),
                    operation.aRowOffset(), operation.aColumnOffset(), operation.bColumnOffset(),
                    operation.columnLength(), operation.resultRowOffset(), operation.bRowOffset(),
                    operation.rowChunkSize(), result.stride(), a.stride(), inputScale.stride(), b.stride(),
                    weightScale.stride());
            return status == Tensor2Native.TENSOR2_OK() ? Either.Right(null) : Either.Left(OpSupport.Unsupported);
        }
        if (result.dType() != DType.F32 || a.dType() != DType.F32) {
            return Either.Left(OpSupport.Unsupported);
        }
        if (b.dType() == DType.BF16) {
            int status = Tensor2Native.tensor2_gemm_f32_bf16(
                    baseSegment(result), baseSegment(a), baseSegment(b),
                    (int) result.shape().first(), operation.aRowOffset(), operation.aColumnOffset(),
                    operation.bColumnOffset(), operation.columnLength(), operation.resultRowOffset(),
                    operation.bRowOffset(), operation.rowChunkSize(), result.stride(), a.stride(), b.stride());
            return status == Tensor2Native.TENSOR2_OK() ? Either.Right(null) : Either.Left(OpSupport.Unsupported);
        }
        TensorRef q8Scale = Q8Layout.scale(b);
        if (b.dType() == DType.I8 && q8Scale != null && q8Aligned(operation)) {
            return batchDotProductF32Q8(operation, q8Scale);
        }
        if (b.dType() != DType.F32) {
            return Either.Left(OpSupport.Unsupported);
        }
        int status = Tensor2Native.tensor2_batch_dot_f32_f32(
                baseSegment(result),
                baseSegment(a),
                baseSegment(b),
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
    public Either<OpSupport, Void> dotProductRows(TensorRef output, TensorRef input, TensorRef weights,
            int inputStart, int inputLength, int weightRowStart, int weightRowCount, int outputColumnStart) {
        return dotProductRows(output, input, weights, inputStart, inputStart, inputLength, weightRowStart,
                weightRowCount, outputColumnStart);
    }

    @Override
    public Either<OpSupport, Void> dotProductBatchChunk(DotProductBatchChunk operation) {
        if (!supportsDotProductBatchChunk(operation)) {
            return Either.Left(OpSupport.Unsupported);
        }
        TensorRef[] results = operation.results();
        TensorRef[] weights = operation.weights();
        TensorRef input = operation.input();
        int status = Tensor2Native.tensor2_dot_product_batch_chunk_i8_q4(
                baseSegment(results[0]), baseSegment(results[1]), baseSegment(input),
                baseSegment(Q8Layout.scale(input)), baseSegment(weights[0]), baseSegment(Q4Layout.scale(weights[0])),
                baseSegment(weights[1]), baseSegment(Q4Layout.scale(weights[1])), results[0].shape().first(),
                operation.inputColumnStart(), operation.weightColumnStart(), operation.columnLength(),
                operation.weightRowStart(), operation.weightRowCount(), operation.outputColumnStart(),
                results[0].stride(), results[1].stride(), input.stride(), Q8Layout.scale(input).stride(),
                weights[0].stride(), Q4Layout.scale(weights[0]).stride(), weights[1].stride(),
                Q4Layout.scale(weights[1]).stride());
        return status == Tensor2Native.TENSOR2_OK() ? Either.Right(null) : Either.Left(OpSupport.Unsupported);
    }

    @Override
    public boolean supportsDotProductBatchChunk(DotProductBatchChunk operation) {
        TensorRef[] results = operation.results();
        TensorRef[] weights = operation.weights();
        return loaded && results != null && weights != null && results.length == 2 && weights.length == 2
                && operation.input() != null && operation.input().dType() == DType.I8
                && results[0].dType() == DType.F32 && results[1].dType() == DType.F32
                && weights[0].dType() == DType.Q4 && weights[1].dType() == DType.Q4
                && Q8Layout.scale(operation.input()) != null && Q4Layout.scale(weights[0]) != null
                && Q4Layout.scale(weights[1]) != null
                && operation.inputColumnStart() % Q8Layout.BLOCK_SIZE == 0
                && operation.weightColumnStart() % Q4Layout.BLOCK_SIZE == 0
                && operation.columnLength() % Q8Layout.BLOCK_SIZE == 0;
    }

    @Override
    public Either<OpSupport, Void> activationMultiplyQuantize(ActivationMultiplyQuantize operation) {
        if (!supportsActivationMultiplyQuantize(operation)) {
            return Either.Left(OpSupport.Unsupported);
        }
        NativeSimd.activation_multiply_quantize_silu_q8(operation.gate().memorySegment(), operation.up().memorySegment(),
                operation.output().memorySegment(), Q8Layout.scale(operation.output()).memorySegment(),
                operation.gate().shape().first(), operation.offset(), operation.length(), operation.gate().stride(),
                operation.up().stride(), operation.output().stride(), Q8Layout.scale(operation.output()).stride());
        return Either.Right(null);
    }

    @Override
    public boolean supportsActivationMultiplyQuantize(ActivationMultiplyQuantize operation) {
        return operation.activation() == io.teknek.deliverance.math.ActivationFunction.Type.SILU
                && operation.qtype() == DType.I8
                && operation.gate().dType() == DType.F32 && operation.up().dType() == DType.F32
                && (operation.output() == null || operation.output().dType() == DType.I8)
                && operation.gate().shape().equals(operation.up().shape())
                && operation.offset() % Q8Layout.BLOCK_SIZE == 0
                && operation.length() % Q8Layout.BLOCK_SIZE == 0
                && (operation.output() == null || Q8Layout.scale(operation.output()) != null);
    }

    @Override
    public Either<OpSupport, Void> dotProductRows(TensorRef output, TensorRef input, TensorRef weights,
            int inputColumnStart, int weightColumnStart, int columnLength, int weightRowStart,
            int weightRowCount, int outputColumnStart) {
        if (!loaded || (output.dType() != DType.F32 && output.dType() != DType.BF16)
                || (input.dType() != DType.F32 && input.dType() != DType.BF16 && input.dType() != DType.I8)) {
            return Either.Left(OpSupport.Unsupported);
        }
        try {
            if (input.dType() == DType.I8 && weights.dType() == DType.Q4) {
                TensorRef inputScales = Q8Layout.scale(input);
                TensorRef weightScales = Q4Layout.scale(weights);
                if (output.dType() != DType.F32 || inputScales == null || weightScales == null
                        || inputColumnStart % Q8Layout.BLOCK_SIZE != 0
                        || weightColumnStart % Q4Layout.BLOCK_SIZE != 0
                        || columnLength % Q8Layout.BLOCK_SIZE != 0) {
                    return Either.Left(OpSupport.Unsupported);
                }
                int status = Tensor2Native.tensor2_dot_product_rows_i8_q4(
                        baseSegment(output), baseSegment(input), baseSegment(inputScales), baseSegment(weights),
                        baseSegment(weightScales), (int) output.shape().first(), inputColumnStart, weightColumnStart,
                        columnLength, weightRowStart, weightRowCount, outputColumnStart, output.stride(), input.stride(),
                        inputScales.stride(), weights.stride(), weightScales.stride());
                return status == Tensor2Native.TENSOR2_OK() ? Either.Right(null) : Either.Left(OpSupport.Unsupported);
            }
            if (input.dType() == DType.F32 && weights.dType() == DType.BF16 && output.dType() == DType.F32) {
                return batchDotProduct(new BatchDotProduct()
                        .result(output).a(input).b(weights)
                        .aColumnOffset(inputColumnStart).bColumnOffset(weightColumnStart)
                        .columnLength(columnLength)
                        .resultRowOffset(outputColumnStart - weightRowStart)
                        .bRowOffset(weightRowStart).rowChunkSize(weightRowCount));
            }
            if (input.dType() == DType.F32 && weights.dType() == DType.Q4 && output.dType() == DType.F32) {
                return batchDotProduct(new BatchDotProduct()
                        .result(output).a(input).b(weights)
                        .aColumnOffset(inputColumnStart).bColumnOffset(weightColumnStart)
                        .columnLength(columnLength)
                        .resultRowOffset(outputColumnStart - weightRowStart)
                        .bRowOffset(weightRowStart).rowChunkSize(weightRowCount));
            }
            if (input.dType() == DType.I8 && weights.dType() == DType.Q4 && output.dType() == DType.F32) {
                return batchDotProduct(new BatchDotProduct()
                        .result(output).a(input).b(weights)
                        .aColumnOffset(inputColumnStart).bColumnOffset(weightColumnStart)
                        .columnLength(columnLength)
                        .resultRowOffset(outputColumnStart - weightRowStart)
                        .bRowOffset(weightRowStart).rowChunkSize(weightRowCount));
            }
            if (input.dType() == DType.I8) {
                return Either.Left(OpSupport.Unsupported);
            }
            if (input.dType() == DType.BF16 && weights.dType() == DType.Q4) {
                TensorRef scales = Q4Layout.scale(weights);
                if (scales == null) {
                    return Either.Left(OpSupport.Unsupported);
                }
                int status = Tensor2Native.tensor2_dot_product_rows_bf16_q4(
                        baseSegment(output), baseSegment(input), baseSegment(weights), baseSegment(scales),
                        output.dType() == DType.BF16 ? 1 : 0, (int) output.shape().first(), inputColumnStart,
                        weightColumnStart, columnLength, weightRowStart, weightRowCount, outputColumnStart,
                        output.stride(), input.stride(), weights.stride(), scales.stride());
                return status == Tensor2Native.TENSOR2_OK() ? Either.Right(null) : Either.Left(OpSupport.Unsupported);
            }
            if (weights.dType() == DType.Q4) {
                TensorRef scales = Q4Layout.scale(weights);
                if (output.dType() != DType.F32 || input.dType() != DType.F32 || scales == null) {
                    return Either.Left(OpSupport.Unsupported);
                }
                int status = Tensor2Native.tensor2_dot_product_rows_f32_q4(
                        baseSegment(output), baseSegment(input), baseSegment(weights), baseSegment(scales),
                        (int) output.shape().first(), inputColumnStart, weightColumnStart, columnLength,
                        weightRowStart, weightRowCount, outputColumnStart, output.stride(), input.stride(),
                        weights.stride(), scales.stride());
                return status == Tensor2Native.TENSOR2_OK() ? Either.Right(null) : Either.Left(OpSupport.Unsupported);
            }
            if (weights.dType() == DType.I8) {
                TensorRef scales = Q8Layout.scale(weights);
                if (output.dType() != DType.F32 || input.dType() != DType.F32 || scales == null
                        || inputColumnStart != weightColumnStart
                        || inputColumnStart % Q8Layout.BLOCK_SIZE != 0
                        || columnLength % Q8Layout.BLOCK_SIZE != 0) {
                    return Either.Left(OpSupport.Unsupported);
                }
                int status = Tensor2Native.tensor2_dot_product_rows_f32_q8(
                        baseSegment(output), baseSegment(input), baseSegment(weights), baseSegment(scales),
                        (int) output.shape().first(), inputColumnStart, columnLength, weightRowStart, weightRowCount,
                        outputColumnStart, output.stride(), input.stride(), weights.stride(), scales.stride());
                return status == Tensor2Native.TENSOR2_OK() ? Either.Right(null) : Either.Left(OpSupport.Unsupported);
            }
            if (weights.dType() != DType.F32) {
                return Either.Left(OpSupport.Unsupported);
            }
            if (output.dType() != DType.F32 || input.dType() != DType.F32) {
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
        } catch (LinkageError | RuntimeException e) {
            return Either.Left(OpSupport.Unsupported);
        }
    }

    @Override
    public Either<OpSupport, Void> saxpy(float alpha, TensorRef x, TensorRef y, int xOffset, int yOffset,
            int length) {
        if (!loaded || x.dType() != DType.F32 || y.dType() != DType.F32 || y.shape().first() != 1) {
            return Either.Left(OpSupport.Unsupported);
        }
        try {
            MemorySegment xSegment = x.memorySegment().asSlice(x.memorySegmentOffset(x.shape().getOffset(0, 0)));
            MemorySegment ySegment = y.memorySegment().asSlice(y.memorySegmentOffset(y.shape().getOffset(0, 0)));
            int status = Tensor2Native.tensor2_saxpy_f32(alpha, xSegment, ySegment,
                    xOffset, yOffset, length);
            if (status != Tensor2Native.TENSOR2_OK()) {
                return Either.Left(OpSupport.Unsupported);
            }
            return Either.Right(null);
        } catch (LinkageError | RuntimeException e) {
            return Either.Left(OpSupport.Unsupported);
        }
    }

    @Override
    public Either<OpSupport, Void> saxpy(TensorRef alpha, TensorRef x, TensorRef y, int xOffset, int yOffset,
            int length, int alphaOffset, int xRowOffset, int batchSize) {
        if (!loaded || alpha.dType() != DType.F32 || x.dType() != DType.F32 || y.dType() != DType.F32
                || y.shape().first() != 1) {
            return Either.Left(OpSupport.Unsupported);
        }
        try {
            MemorySegment alphaSegment = alpha.memorySegment().asSlice(
                    alpha.memorySegmentOffset(alpha.shape().getOffset(0, 0)));
            MemorySegment xSegment = x.memorySegment().asSlice(x.memorySegmentOffset(x.shape().getOffset(0, 0)));
            MemorySegment ySegment = y.memorySegment().asSlice(y.memorySegmentOffset(y.shape().getOffset(0, 0)));
            int status = Tensor2Native.tensor2_saxpy_f32_batch(alphaSegment, xSegment,
                    ySegment, xOffset, yOffset, length, alphaOffset, xRowOffset, batchSize,
                    x.stride());
            if (status != Tensor2Native.TENSOR2_OK()) {
                return Either.Left(OpSupport.Unsupported);
            }
            return Either.Right(null);
        } catch (LinkageError | RuntimeException e) {
            return Either.Left(OpSupport.Unsupported);
        }
    }

    @Override
    public Either<OpSupport, Void> scale(float factor, TensorRef target, int offset, int length) {
        if (!loaded) {
            return Either.Left(OpSupport.Unsupported);
        }
        try {
            int status;
            if (target.dType() == DType.F32) {
                status = Tensor2Native.tensor2_scale_f32(baseSegment(target), factor,
                        (int) target.shape().first(), offset, length, target.stride());
            } else if (target.dType() == DType.BF16) {
                status = Tensor2Native.tensor2_scale_bf16(baseSegment(target), factor,
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

    private static MemorySegment baseSegment(TensorRef tensor) {
        return tensor.memorySegment().asSlice(tensor.memorySegmentOffset(tensor.shape().getOffset(0, 0)));
    }
}
