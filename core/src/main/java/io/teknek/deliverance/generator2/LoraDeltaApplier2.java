package io.teknek.deliverance.generator2;

import io.teknek.deliverance.generator.ForwardPhase;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.safetensors.LoraLayerDelta;
import io.teknek.deliverance.tensor2.BatchDotProduct;
import io.teknek.deliverance.tensor2.Accumulate;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;

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
        model.runChunks(metricName, 0, outputLength, model.primaryTensorOperations().parallelSplitSize(),
                java.util.Optional.empty(), (chunkStart, chunkSize) -> {
                    if (phase == ForwardPhase.PREFILL && input.shape().first() > 1) {
                        lighter.batchDotProduct(new BatchDotProduct()
                                .result(output).a(input).b(weights)
                                .aColumnOffset(0).bColumnOffset(0).columnLength(inputLength)
                                .resultRowOffset(0).bRowOffset(chunkStart).rowChunkSize(chunkSize));
                    } else {
                        lighter.dotProductRows(output, input, weights, 0, inputLength, chunkStart, chunkSize,
                                chunkStart);
                    }
                });
    }
}
