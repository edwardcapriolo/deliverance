package io.teknek.deliverance.tensor.operations.gpunative;

import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;

/** Dedicated downcall binding for the Tensor2 Q4 projection ABI. */
public final class Tensor2GpuNative {
    private Tensor2GpuNative() {
    }

    private static final FunctionDescriptor BATCH_DOT = FunctionDescriptor.ofVoid(
            NativeGPU.C_LONG, NativeGPU.C_LONG, NativeGPU.C_POINTER, NativeGPU.C_POINTER,
            NativeGPU.C_INT, NativeGPU.C_INT, NativeGPU.C_LONG, NativeGPU.C_LONG,
            NativeGPU.C_INT, NativeGPU.C_INT, NativeGPU.C_POINTER, NativeGPU.C_INT,
            NativeGPU.C_INT, NativeGPU.C_INT, NativeGPU.C_INT, NativeGPU.C_INT,
            NativeGPU.C_INT, NativeGPU.C_INT, NativeGPU.C_INT, NativeGPU.C_INT);
    private static final MethodHandle BATCH_DOT_HANDLE = Linker.nativeLinker().downcallHandle(
            NativeGPU.findOrThrow("tensor2_gpu_batch_dot_i8_q4"), BATCH_DOT);
    private static final FunctionDescriptor F32_BATCH_DOT = FunctionDescriptor.ofVoid(
            NativeGPU.C_LONG, NativeGPU.C_LONG, NativeGPU.C_POINTER, NativeGPU.C_INT, NativeGPU.C_INT,
            NativeGPU.C_LONG, NativeGPU.C_LONG, NativeGPU.C_INT, NativeGPU.C_INT, NativeGPU.C_POINTER,
            NativeGPU.C_INT, NativeGPU.C_INT, NativeGPU.C_INT, NativeGPU.C_INT, NativeGPU.C_INT,
            NativeGPU.C_INT, NativeGPU.C_INT, NativeGPU.C_INT, NativeGPU.C_INT);
    private static final MethodHandle F32_BATCH_DOT_HANDLE = Linker.nativeLinker().downcallHandle(
            NativeGPU.findOrThrow("tensor2_gpu_batch_dot_f32_q4"), F32_BATCH_DOT);
    private static final FunctionDescriptor DEVICE_BATCH_DOT = FunctionDescriptor.ofVoid(
            NativeGPU.C_LONG, NativeGPU.C_LONG, NativeGPU.C_LONG, NativeGPU.C_LONG,
            NativeGPU.C_INT, NativeGPU.C_INT, NativeGPU.C_LONG, NativeGPU.C_LONG,
            NativeGPU.C_INT, NativeGPU.C_INT, NativeGPU.C_LONG, NativeGPU.C_INT,
            NativeGPU.C_INT, NativeGPU.C_INT, NativeGPU.C_INT, NativeGPU.C_INT,
            NativeGPU.C_INT, NativeGPU.C_INT, NativeGPU.C_INT, NativeGPU.C_INT);
    private static final MethodHandle DEVICE_I8_HANDLE = Linker.nativeLinker().downcallHandle(
            NativeGPU.findOrThrow("tensor2_gpu_batch_dot_i8_q4_device"), DEVICE_BATCH_DOT);
    private static final MethodHandle DEVICE_F32_HANDLE = Linker.nativeLinker().downcallHandle(
            NativeGPU.findOrThrow("tensor2_gpu_batch_dot_f32_q4_device"), DEVICE_BATCH_DOT);

    public static void batchDotI8Q4(long scratchId, long shaderId, MemorySegment input,
            MemorySegment inputScales, int inputOffset, int inputLimit, long weights,
            long weightScales, int weightOffset, int weightLimit, MemorySegment result,
            int resultOffset, int resultLimit, int resultRows, int weightRowOffset,
            int weightRows, int columns, int inputStride, int weightStride, int resultStride) {
        try {
            BATCH_DOT_HANDLE.invokeExact(scratchId, shaderId, input, inputScales, inputOffset, inputLimit,
                    weights, weightScales, weightOffset, weightLimit, result, resultOffset, resultLimit,
                    resultRows, weightRowOffset, weightRows, columns, inputStride, weightStride, resultStride);
        } catch (Throwable e) {
            throw new AssertionError("Tensor2 GPU downcall failed", e);
        }
    }

    public static void batchDotF32Q4(long scratchId, long shaderId, MemorySegment input,
            int inputOffset, int inputLimit, long weights, long weightScales, int weightOffset,
            int weightLimit, MemorySegment result, int resultOffset, int resultLimit, int resultRows,
            int weightRowOffset, int weightRows, int columns, int inputStride, int weightStride,
            int resultStride) {
        try {
            F32_BATCH_DOT_HANDLE.invokeExact(scratchId, shaderId, input, inputOffset, inputLimit, weights,
                    weightScales, weightOffset, weightLimit, result, resultOffset, resultLimit, resultRows,
                    weightRowOffset, weightRows, columns, inputStride, weightStride, resultStride);
        } catch (Throwable e) {
            throw new AssertionError("Tensor2 GPU downcall failed", e);
        }
    }

    public static void batchDotI8Q4Device(long scratchId, long shaderId, long input, long inputScales,
            int inputOffset, int inputLimit, long weights, long weightScales, int weightOffset, int weightLimit,
            long result, int resultOffset, int resultLimit, int resultRows, int weightRowOffset, int weightRows,
            int columns, int inputStride, int weightStride, int resultStride) {
        invokeDevice(DEVICE_I8_HANDLE, scratchId, shaderId, input, inputScales, inputOffset, inputLimit, weights,
                weightScales, weightOffset, weightLimit, result, resultOffset, resultLimit, resultRows,
                weightRowOffset, weightRows, columns, inputStride, weightStride, resultStride);
    }

    public static void batchDotF32Q4Device(long scratchId, long shaderId, long input, long inputScales,
            int inputOffset, int inputLimit, long weights, long weightScales, int weightOffset, int weightLimit,
            long result, int resultOffset, int resultLimit, int resultRows, int weightRowOffset, int weightRows,
            int columns, int inputStride, int weightStride, int resultStride) {
        invokeDevice(DEVICE_F32_HANDLE, scratchId, shaderId, input, inputScales, inputOffset, inputLimit, weights,
                weightScales, weightOffset, weightLimit, result, resultOffset, resultLimit, resultRows,
                weightRowOffset, weightRows, columns, inputStride, weightStride, resultStride);
    }

    private static void invokeDevice(MethodHandle handle, long scratchId, long shaderId, long input, long inputScales,
            int inputOffset, int inputLimit, long weights, long weightScales, int weightOffset, int weightLimit,
            long result, int resultOffset, int resultLimit, int resultRows, int weightRowOffset, int weightRows,
            int columns, int inputStride, int weightStride, int resultStride) {
        try {
            handle.invokeExact(scratchId, shaderId, input, inputScales, inputOffset, inputLimit, weights,
                    weightScales, weightOffset, weightLimit, result, resultOffset, resultLimit, resultRows,
                    weightRowOffset, weightRows, columns, inputStride, weightStride, resultStride);
        } catch (Throwable e) {
            throw new AssertionError("Tensor2 GPU device downcall failed", e);
        }
    }
}
