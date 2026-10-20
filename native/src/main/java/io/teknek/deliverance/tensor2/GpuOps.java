package io.teknek.deliverance.tensor2;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.operations.gpunative.NativeGPU;
import io.teknek.dysfx.Either;

import java.io.IOException;
import java.io.InputStream;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.util.concurrent.atomic.AtomicBoolean;

/** GPU tensor2 operations backed by dedicated WebGPU compute pipelines. */
public final class GpuOps implements TensorOps, AutoCloseable {
    private final long scaleShader;
    private final AtomicBoolean closed = new AtomicBoolean();

    public GpuOps() {
        this.scaleShader = registerScaleShader();
    }

    @Override
    public Either<OpSupport, Void> scale(float factor, TensorRef target, int offset, int length) {
        if (closed.get()) {
            throw new IllegalStateException("GpuOps is closed");
        }
        if (!"gpu".equals(target.device()) || target.dType() != DType.F32 || target.dims() != 2
                || offset < 0 || length < 0
                || offset + length > target.shape().last()) {
            return Either.Left(OpSupport.Unsupported);
        }
        if (length == 0) {
            return Either.Right(null);
        }
        NativeGPU.gpu_scale_f32(scaleShader, target.deviceHandle(), factor, offset, length,
                (int) target.shape().first(), (int) target.shape().last());
        return Either.Right(null);
    }

    @Override
    public Either<OpSupport, Void> multiplyAccumulate(TensorRef a, TensorRef b, int offset, int length) {
        return Either.Left(OpSupport.Unsupported);
    }

    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            NativeGPU.free_scale_shader(scaleShader);
        }
    }

    private static long registerScaleShader() {
        try (InputStream input = GpuOps.class.getResourceAsStream("/scale_f32.wgsl")) {
            if (input == null) {
                throw new IllegalStateException("Missing scale_f32.wgsl resource");
            }
            byte[] source = input.readAllBytes();
            ByteBuffer buffer = ByteBuffer.allocateDirect(source.length);
            buffer.put(source).flip();
            long id = NativeGPU.register_scale_shader(MemorySegment.ofBuffer(buffer), source.length);
            if (id == -1L) {
                throw new IllegalStateException("Failed to register scale_f32.wgsl");
            }
            return id;
        } catch (IOException e) {
            throw new IllegalStateException("Failed to load scale_f32.wgsl", e);
        }
    }
}
