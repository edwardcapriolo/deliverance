package io.teknek.reckoning.generator2;

import io.teknek.reckoning.DType;
import io.teknek.reckoning.tensor.TensorShape;
import io.teknek.reckoning.tensor2.BatchDotProduct;
import io.teknek.reckoning.tensor2.Lighter;
import io.teknek.reckoning.tensor2.TensorProviderKind;
import io.teknek.reckoning.tensor2.TensorRef;

import java.util.IdentityHashMap;
import java.util.Map;

/** Coordinates the CPU/GPU boundary for one-token decode projections. */
public final class GpuDecodeExecutionContext implements AutoCloseable {
    private final Lighter lighter;
    private final Map<TensorRef, TensorRef> residentWeights = new IdentityHashMap<>();

    public GpuDecodeExecutionContext(Lighter lighter) {
        this.lighter = java.util.Objects.requireNonNull(lighter, "lighter");
    }

    public synchronized boolean project(TensorRef input, TensorRef output, TensorRef weights, int inputLength,
            int outputOffset, int outputLength) {
        if (!lighter.tensorOperations().containsKey(TensorProviderKind.GPU)
                || input.shape().first() != 1 || output.shape().first() != 1
                || output.dType() != DType.F32) {
            return false;
        }
        TensorRef gpuInput = null;
        TensorRef gpuOutput = null;
        try {
            TensorRef gpuWeights = residentWeight(weights);
            gpuInput = lighter.to(input, "gpu");
            gpuOutput = lighter.allocate(DType.F32, TensorShape.of(1, output.shape().last()), "gpu");
            lighter.batchDotProduct(new BatchDotProduct()
                    .result(gpuOutput)
                    .a(gpuInput)
                    .b(gpuWeights)
                    .aRowOffset(0)
                    .aColumnOffset(0)
                    .bColumnOffset(0)
                    .columnLength(inputLength)
                    .resultRowOffset(0)
                    .bRowOffset(outputOffset)
                    .rowChunkSize(outputLength));
            try (TensorRef cpuOutput = lighter.to(gpuOutput, "cpu")) {
                lighter.copy(cpuOutput, outputOffset, output, outputOffset, outputLength);
            }
            return true;
        } catch (RuntimeException | Error failure) {
            return false;
        } finally {
            if (gpuInput != null) {
                gpuInput.close();
            }
            if (gpuOutput != null) {
                gpuOutput.close();
            }
        }
    }

    @Override
    public synchronized void close() {
        residentWeights.values().forEach(TensorRef::close);
        residentWeights.clear();
    }

    private TensorRef residentWeight(TensorRef weight) {
        if ("gpu".equals(weight.device())) {
            return weight;
        }
        return residentWeights.computeIfAbsent(weight, source -> lighter.to(source, "gpu"));
    }
}
