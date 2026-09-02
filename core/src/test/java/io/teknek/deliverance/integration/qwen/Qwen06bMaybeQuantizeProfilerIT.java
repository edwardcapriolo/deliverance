package io.teknek.deliverance.integration.qwen;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.model.DoNothingGenerateEvent;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.operations.NaiveTensorOperations;
import io.teknek.deliverance.tensor.operations.PanamaTensorOperations;
import io.teknek.deliverance.tensorlib.TensorPlan;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

@Tag("small-model")
public class Qwen06bMaybeQuantizeProfilerIT {

    @Test
    public void aplan(){
        //TensorPlan tp = new TensorPlan(new NaiveTensorOperations(), new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores()));
        //TensorPlan.ImmutableTensor something = tp.immutable("something", null);
        //TensorPlan.Tensor mutable = tp.mutable("abc", null);
        //TensorPlan.Tensor scaled = mutable.scale(50);
        //AbstractTensor scaledT = scaled.materialize();

        //TensorPlan stage1 = new TensorPlan(new NaiveTensorOperations(), new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores()));
        //stage1.input(scaled, scaledT);
    }
    @Test
    public void oneCaseHitsReadOnlyMaybeQuantizePaths() {
        boolean previousProfiling = InferenceProfiler.isEnabled();
        InferenceProfiler.setEnabled(true);
        ModelFetcher fetch = new ModelFetcher("edwardcapriolo", "Qwen3-0.6B-JQ4").withDownload(false);
        Assumptions.assumeTrue(fetch.pathForModel().toFile().isDirectory(),
                "Qwen3-0.6B-JQ4 cache is not present: " + fetch.pathForModel());

        try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetch)
                .withDownload(false)
                .withTensorPlanTrace(true)
                .buildLocalTransformerModel()) {
            InferenceProfiler.reset();
            Response response = model.generate(UUID.randomUUID(), PromptContext.of("What is 2+2? Answer briefly."),
                    new GeneratorParameters().withTemperature(0.0f).withMaxTokens(16).withSeed(42),
                    new DoNothingGenerateEvent());
            InferenceProfiler.printSummary("qwen06b maybeQuantize one-case", 20);
            model.getMetricRegistry().getCounters().entrySet().stream()
                    .filter(entry -> InferenceProfiler.shouldPrintCounter(entry.getKey()))
                    .forEach(entry -> System.out.println("[profile-counter] " + InferenceProfiler.displayName(entry.getKey())
                            + " count=" + InferenceProfiler.counterValue(entry.getKey())));

            assertFalse(response.responseTextWithSpecialTokens.isBlank());
            assertCounterHit("transformerblock.maybe_quantize.pre_attention.copy_or_quantize");
            assertCounterHit("transformerblock.maybe_quantize.pre_ff.copy_or_quantize");
            assertCounterHit("causalselfattention.maybe_quantize.output_projection.copy_or_quantize");
        } finally {
            InferenceProfiler.setEnabled(previousProfiling);
        }
    }

    @Test
    public void naturalBf16QwenHitsReadOnlyMaybeQuantizePaths() {
        boolean previousProfiling = InferenceProfiler.isEnabled();
        InferenceProfiler.setEnabled(true);
        ModelFetcher fetch = new ModelFetcher("Qwen", "Qwen3-0.6B").withDownload(false);
        Assumptions.assumeTrue(fetch.pathForModel().toFile().isDirectory(),
                "Qwen3-0.6B cache is not present: " + fetch.pathForModel());

        try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetch)
                .withDownload(false)
                .withWorkingMemoryType(DType.BF16)
                .withWorkingQuantType(DType.BF16)
                .buildLocalTransformerModel()) {
            InferenceProfiler.reset();
            Response response = model.generate(UUID.randomUUID(), PromptContext.of("What is 2+2? Answer briefly."),
                    new GeneratorParameters().withTemperature(0.0f).withMaxTokens(16).withSeed(42),
                    new DoNothingGenerateEvent());
            InferenceProfiler.printSummary("qwen06b bf16 maybeQuantize one-case", 20);
            model.getMetricRegistry().getCounters().entrySet().stream()
                    .filter(entry -> InferenceProfiler.shouldPrintCounter(entry.getKey()))
                    .forEach(entry -> System.out.println("[profile-counter] " + InferenceProfiler.displayName(entry.getKey())
                            + " count=" + InferenceProfiler.counterValue(entry.getKey())));

            assertFalse(response.responseTextWithSpecialTokens.isBlank());
            assertCounterHit("transformerblock.maybe_quantize.pre_attention.read_only");
            assertCounterHit("transformerblock.maybe_quantize.pre_ff.read_only");
            assertCounterHit("causalselfattention.maybe_quantize.output_projection.read_only");
        } finally {
            InferenceProfiler.setEnabled(previousProfiling);
        }
    }

    private static void assertCounterHit(String name) {
        assertTrue(InferenceProfiler.counterValue(name) > 0, name + " should be hit");
    }
}
