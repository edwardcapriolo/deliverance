package io.teknek.deliverance.tensor2;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.operations.RuntimeSupport;
import io.teknek.deliverance.tensor.operations.gpunative.NativeGPU;
import io.teknek.deliverance.tensor.operations.gpunative.Tensor2GpuNative;
import io.teknek.deliverance.tensor.operations.util.JarSupport;
import io.teknek.dysfx.Either;

import java.io.IOException;
import java.io.InputStream;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.LongBuffer;
import java.util.concurrent.atomic.AtomicLong;

/** Tensor2 GPU provider for the Nemotron F32/I8 x Q4 block projection. */
public final class GpuOps implements TensorOps, AutoCloseable {
    private static final int INPUT_SCRATCH_SIZE = 1 << 24;
    private static final int RESULT_SCRATCH_SIZE = 1 << 24;
    private static final int PARAMS_SIZE = 8 * Integer.BYTES;
    private static final AtomicLong GPU_CALLS = new AtomicLong();
    private static volatile RuntimeState runtime;
    private static volatile Throwable unavailable;

    public GpuOps() {
        state();
    }

    public static boolean isAvailable() {
        try {
            state();
            return true;
        } catch (Throwable ignored) {
            return false;
        }
    }

    /** Counts successful Tensor2 GPU submissions, not provider dispatch attempts. */
    public static long gpuCalls() {
        return GPU_CALLS.get();
    }

    @Override
    public Either<OpSupport, Void> batchDotProduct(BatchDotProduct operation) {
        if (operation.result().dType() != DType.F32 || operation.a().dType() != DType.I8
                && operation.a().dType() != DType.F32 || operation.b().dType() != DType.Q4 || operation.a().dims() != 2
                || operation.b().dims() != 2 || operation.result().dims() != 2
                || operation.kernel() != GemmKernel.BATCH
                || !"gpu".equals(operation.a().device()) || !"gpu".equals(operation.b().device())
                || !"gpu".equals(operation.result().device())
                || !aligned(operation) || operation.rowChunkSize() <= 0) {
            return Either.Left(OpSupport.Unsupported);
        }
        TensorRef inputScale = operation.a().dType() == DType.I8 ? Q8Layout.scale(operation.a()) : null;
        TensorRef weightScale = Q4Layout.scale(operation.b());
        if (operation.a().dType() == DType.I8 && (inputScale == null || !"gpu".equals(inputScale.device()))
                || weightScale == null || !"gpu".equals(weightScale.device())) {
            return Either.Left(OpSupport.Unsupported);
        }

        synchronized (GpuOps.class) {
            RuntimeState state = state();
            int m = operation.result().shape().first();
            int n = operation.rowChunkSize();
            int aOffset = byteOffset(operation.a(), operation.aRowOffset(), operation.aColumnOffset());
            int aLimit = byteOffset(operation.a(), operation.aRowOffset() + m, operation.aColumnOffset());
            int bOffset = byteOffset(operation.b(), operation.bRowOffset(), operation.bColumnOffset());
            int bLimit = byteOffset(operation.b(), operation.bRowOffset() + n, operation.bColumnOffset());
            int resultOffset = operation.result().shape().sparseColumnOffset()
                    - operation.b().shape().sparseRowOffset() - operation.resultRowOffset();
            int resultLimit = Math.toIntExact(operation.result().shape().size() * (long) DType.F32.size());
            try {
                if (operation.a().dType() == DType.I8) {
                    Tensor2GpuNative.batchDotI8Q4Device(state.scratchId, state.i8ShaderId,
                            operation.a().deviceHandle(), inputScale.deviceHandle(), aOffset, aLimit,
                            operation.b().deviceHandle(), weightScale.deviceHandle(), bOffset, bLimit,
                            operation.result().deviceHandle(), resultOffset, resultLimit, m,
                            operation.bRowOffset(), n, operation.columnLength(), operation.a().stride(),
                            operation.b().stride(), operation.result().stride());
                } else {
                    Tensor2GpuNative.batchDotF32Q4Device(state.scratchId, state.f32ShaderId,
                            operation.a().deviceHandle(), 0L, aOffset, aLimit,
                            operation.b().deviceHandle(), weightScale.deviceHandle(), bOffset, bLimit,
                            operation.result().deviceHandle(), resultOffset, resultLimit, m,
                            operation.bRowOffset(), n, operation.columnLength(),
                            operation.a().stride(), operation.b().stride(), operation.result().stride());
                }
                GPU_CALLS.incrementAndGet();
                return Either.Right(null);
            } catch (LinkageError | RuntimeException e) {
                return Either.Left(OpSupport.Unsupported);
            }
        }
    }

    @Override
    public Either<OpSupport, Void> multiplyAccumulate(TensorRef a, TensorRef b, int offset, int length) {
        return Either.Left(OpSupport.Unsupported);
    }

    @Override
    public Either<OpSupport, Void> scale(float factor, TensorRef target, int offset, int length) {
        if (target == null || target.dType() != DType.F32 || target.dims() != 2
                || !"gpu".equals(target.device()) || target.deviceHandle() == 0L
                || offset < 0 || length <= 0 || offset > target.shape().last() - length
                || !Float.isFinite(factor)) {
            return Either.Left(OpSupport.Unsupported);
        }
        synchronized (GpuOps.class) {
            try {
                RuntimeState state = state();
                NativeGPU.tensor2_gpu_scale_f32_device(state.scaleShaderId, target.deviceHandle(), factor,
                        offset, length, target.shape().first(), target.shape().last());
                GPU_CALLS.incrementAndGet();
                return Either.Right(null);
            } catch (LinkageError | RuntimeException e) {
                return Either.Left(OpSupport.Unsupported);
            }
        }
    }

    @Override
    public void close() {
        // Runtime resources are process-wide because provider instances are created per Lighter.
    }

    private static boolean aligned(BatchDotProduct operation) {
        return operation.aColumnOffset() % Q8Layout.BLOCK_SIZE == 0
                && operation.bColumnOffset() % Q4Layout.BLOCK_SIZE == 0
                && operation.columnLength() % Q4Layout.BLOCK_SIZE == 0;
    }

    private static int byteOffset(TensorRef tensor, int row, int column) {
        long offset = tensor.shape().getOffset(row, column);
        return Math.toIntExact(tensor.dType() == DType.Q4 ? offset / 2L : offset * tensor.dType().size());
    }

    private static RuntimeState state() {
        RuntimeState current = runtime;
        if (current != null) return current;
        if (unavailable != null) throw new IllegalStateException("Tensor2 GPU unavailable", unavailable);
        synchronized (GpuOps.class) {
            if (runtime != null) return runtime;
            try {
                loadLibraries();
                ByteBuffer result = ByteBuffer.allocateDirect(Long.BYTES * 3);
                LongBuffer values = result.asLongBuffer();
                NativeGPU.init_gpu(MemorySegment.ofBuffer(result));
                if (values.get(0) < 0) throw new IllegalStateException("No GPU adapter");
                long i8Shader = registerShader("/gemm_i8q4.wgsl");
                long f32Shader = registerShader("/gemm_q4.wgsl");
                long scaleShader = registerScaleShader("/scale_f32.wgsl");
                 long scratch = NativeGPU.register_scratch_buffers(PARAMS_SIZE, INPUT_SCRATCH_SIZE,
                        RESULT_SCRATCH_SIZE);
                if (scratch < 0) throw new IllegalStateException("GPU scratch allocation failed");
                runtime = new RuntimeState(scratch, i8Shader, f32Shader, scaleShader);
                return runtime;
            } catch (Throwable t) {
                unavailable = t;
                throw new IllegalStateException("Tensor2 GPU unavailable", t);
            }
        }
    }

    private static void loadLibraries() {
        if (RuntimeSupport.isWin()) {
            if (!JarSupport.maybeLoadLibrary("dxcompiler")) System.loadLibrary("dxcompiler");
            if (!JarSupport.maybeLoadLibrary("dxil")) System.loadLibrary("dxil");
        }
        if (!JarSupport.maybeLoadLibrary("webgpu_dawn")) System.loadLibrary("webgpu_dawn");
        if (!JarSupport.maybeLoadLibrary("deliverancegpu")) System.loadLibrary("deliverancegpu");
    }

    private static long registerShader(String resource) throws IOException {
        try (InputStream input = GpuOps.class.getResourceAsStream(resource)) {
            if (input == null) throw new IOException("Missing GPU shader " + resource);
            byte[] source = input.readAllBytes();
            ByteBuffer buffer = ByteBuffer.allocateDirect(source.length);
            buffer.put(source).flip();
            long id = NativeGPU.register_shader(MemorySegment.ofBuffer(buffer), source.length);
            if (id < 0) throw new IllegalStateException("GPU shader registration failed");
            return id;
        }
    }

    private static long registerScaleShader(String resource) throws IOException {
        try (InputStream input = GpuOps.class.getResourceAsStream(resource)) {
            if (input == null) throw new IOException("Missing GPU shader " + resource);
            byte[] source = input.readAllBytes();
            ByteBuffer buffer = ByteBuffer.allocateDirect(source.length);
            buffer.put(source).flip();
            long id = NativeGPU.tensor2_register_scale_shader(MemorySegment.ofBuffer(buffer), source.length);
            if (id < 0) throw new IllegalStateException("GPU scale shader registration failed");
            return id;
        }
    }

    private record RuntimeState(long scratchId, long i8ShaderId, long f32ShaderId, long scaleShaderId) {
    }
}
