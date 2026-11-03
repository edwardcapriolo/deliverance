package io.teknek.reckoning.generator2;

import io.teknek.reckoning.generator.ForwardPhase;
import io.teknek.reckoning.model.AbstractModel;
import io.teknek.reckoning.model.InferenceProfiler;
import io.teknek.reckoning.safetensors.LoraLayerDelta;
import io.teknek.reckoning.tensor2.BatchDotProduct;
import io.teknek.reckoning.tensor2.Accumulate;
import io.teknek.reckoning.tensor2.Lighter;
import io.teknek.reckoning.tensor2.TensorRef;

/** Applies a resolved LoRA delta using TensorRef operations. */
final class LoraDeltaApplier2 {
    private LoraDeltaApplier2() {
    }

    static void apply(AbstractModel model, Lighter lighter, TensorRef output, TensorRef input,
            LoraLayerDelta delta, ForwardPhase phase, String metricName) {
        InferenceProfiler.counter(model.getMetricRegistry(), "lora.delta.ref.apply").inc();
        try (TensorRef loraA = TensorRef.borrowed(delta.loraA());
             TensorRef scaledLoraB = TensorRef.borrowed(delta.scaledLoraB())) {
            TensorRef denseInput = input;
            boolean closeDenseInput = false;
            if (input.dType() != loraA.dType()) {
                denseInput = lighter.reshape(input, loraA.dType());
                closeDenseInput = true;
            }
            try (TensorRef rankResult = model.makeDenseTensorRef(input.shape().first(), delta.rank());
                 TensorRef deltaResult = model.makeDenseTensorRef(input.shape().first(), output.shape().last())) {
                project(model, lighter, rankResult, denseInput, loraA, phase, metricName + ".a");
                project(model, lighter, deltaResult, rankResult, scaledLoraB, phase, metricName + ".b");
                lighter.accumulate(new Accumulate(deltaResult).into(output)
                        .offsetAndLength(0, output.shape().last()));
            } finally {
                if (closeDenseInput) {
                    denseInput.close();
                }
            }
        }
    }

    private static void project(AbstractModel model, Lighter lighter, TensorRef output, TensorRef input,
            TensorRef weights, ForwardPhase phase, String metricName) {
        int inputLength = input.shape().last();
        int outputLength = output.shape().last();
        model.runChunks(metricName, 0, outputLength, 32,
                java.util.Optional.empty(), (chunkStart, chunkSize) -> {
                    if (phase == ForwardPhase.PREFILL && input.shape().first() > 1) {
                        lighter.batchDotProduct(new BatchDotProduct()
                                .result(output).a(input).b(weights)
                                .aColumnOffset(0).bColumnOffset(0).columnLength(inputLength)
                                .resultRowOffset(0).bRowOffset(chunkStart).rowChunkSize(chunkSize)
                                .threading(io.teknek.reckoning.tensor2.Threading.NONE));
                    } else {
                        lighter.dotProductRows(output, input, weights, 0, 0, inputLength, chunkStart, chunkSize,
                                chunkStart, io.teknek.reckoning.tensor2.GemmKernel.ROW,
                                io.teknek.reckoning.tensor2.Threading.NONE);
                    }
                });
    }
}
