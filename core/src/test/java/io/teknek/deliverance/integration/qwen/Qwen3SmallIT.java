package io.teknek.deliverance.integration.qwen;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.model.DoNothingGenerateEvent;
import io.teknek.deliverance.model.GenerateEvent;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.model.TensorProviderKind;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.tensorlib.TensorRuntimeMode;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

@Tag("large-model")
public class Qwen3SmallIT {

    @Test
    public void qwen306BLoadsAndGeneratesShortAnswer() {
        ModelFetcher fetch = new ModelFetcher("Qwen", "Qwen3-0.6B");
        try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetch)
                .withTensorRuntimeMode(TensorRuntimeMode.ENFORCE)
                .withGpuDecodeAttention(true)
                .buildLocalTransformerModel()) {
            PromptContext prompt = model.promptSupport().orElseThrow().builder()
                    .addTemplateArgs(Map.of("enable_thinking", false))
                    .addUserMessage("What is 1 + 1? Answer with only the number.")
                    .build();
            Response response = model.generate(UUID.randomUUID(), prompt,
                    new GeneratorParameters().withTemperature(0.7f).withTopP(0.8f).withTopK(20.0f).withMaxTokens(8),
                    new DoNothingGenerateEvent());
            System.out.println("QWEN3_06B_SHORT=" + response.responseTextWithSpecialTokens.replace("\n", "\\n"));
            printTensorRuntimeCounters(model.getMetricRegistry());
            assertFalse(response.responseTextWithSpecialTokens.isBlank());
        }
    }

    @Test
    public void qwen306BGpuDecodeAttentionShort() {
        boolean previousProfiling = InferenceProfiler.isEnabled();
        InferenceProfiler.setEnabled(true);
        InferenceProfiler.reset();
        ModelFetcher fetch = new ModelFetcher("Qwen", "Qwen3-0.6B");
        try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetch)
                .withTensorRuntimeMode(TensorRuntimeMode.ENFORCE)
                .withGpuDecodeAttention(true)
                .buildLocalTransformerModel()) {
            assertTrue(model.tensorOperations(TensorProviderKind.GPU).isPresent(),
                    "GPU tensor operations must be available for this integration test");
            PromptContext prompt = model.promptSupport().orElseThrow().builder()
                    .addTemplateArgs(Map.of("enable_thinking", false))
                    .addUserMessage("What is 1 + 1? Answer with only the number.")
                    .build();
            Response response = model.generate(UUID.randomUUID(), prompt,
                    new GeneratorParameters().withTemperature(0.7f).withTopP(0.8f).withTopK(20.0f).withMaxTokens(8),
                    new DoNothingGenerateEvent());
            System.out.println("QWEN3_06B_SHORT_GPU_DECODE_ATTENTION="
                    + response.responseTextWithSpecialTokens.replace("\n", "\\n"));
            printTensorRuntimeCounters(model.getMetricRegistry());
            printCounter(model.getMetricRegistry(),
                    "causalselfattention.decode_paged_attention.provider_Native_GPU_Operations");
            printCounter(model.getMetricRegistry(),
                    "kvcacheselfattention.decode_paged_attention.provider_Native_GPU_Operations");
            assertFalse(response.responseTextWithSpecialTokens.isBlank());
            assertTrue(gpuDecodePagedAttentionCount(model) > 0,
                    "decode paged attention should use Native GPU Operations");
        } finally {
            InferenceProfiler.setEnabled(previousProfiling);
        }
    }

    @Test
    public void qwen34BJq4GpuDecodeAttentionShort() {
        boolean previousProfiling = InferenceProfiler.isEnabled();
        InferenceProfiler.setEnabled(true);
        InferenceProfiler.reset();
        ModelFetcher fetch = new ModelFetcher("edwardcapriolo", "Qwen3-4B-JQ4");
        try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetch)
                .withTensorRuntimeMode(TensorRuntimeMode.ANALYZE)
                .withGpuDecodeAttention(true)
                .buildLocalTransformerModel()) {
            assertTrue(model.tensorOperations(TensorProviderKind.GPU).isPresent(),
                    "GPU tensor operations must be available for this integration test");
            PromptContext prompt = model.promptSupport().orElseThrow().builder()
                    .addTemplateArgs(Map.of("enable_thinking", false))
                    .addUserMessage("What is 1 + 1? Answer with only the number.")
                    .build();
            Response response = model.generate(UUID.randomUUID(), prompt,
                    new GeneratorParameters().withTemperature(0.7f).withTopP(0.8f).withTopK(20.0f).withMaxTokens(8),
                    new DoNothingGenerateEvent());
            System.out.println("QWEN3_4B_JQ4_SHORT_GPU_DECODE_ATTENTION="
                    + response.responseTextWithSpecialTokens.replace("\n", "\\n"));
            printTensorRuntimeCounters(model.getMetricRegistry());
            printCounter(model.getMetricRegistry(),
                    "causalselfattention.decode_paged_attention.provider_Native_GPU_Operations");
            printCounter(model.getMetricRegistry(),
                    "kvcacheselfattention.decode_paged_attention.provider_Native_GPU_Operations");
            assertFalse(response.responseTextWithSpecialTokens.isBlank());
            assertTrue(gpuDecodePagedAttentionCount(model) > 0,
                    "decode paged attention should use Native GPU Operations");
        } finally {
            InferenceProfiler.setEnabled(previousProfiling);
        }
    }

    @Disabled("gpu code is very slow here")
    public void qwen34BJq4GpuDecodeAttentionLong() {
        boolean previousProfiling = InferenceProfiler.isEnabled();
        InferenceProfiler.setEnabled(true);
        InferenceProfiler.reset();
        ModelFetcher fetch = new ModelFetcher("edwardcapriolo", "Qwen3-4B-JQ4");
        try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetch)
                .withTensorRuntimeMode(TensorRuntimeMode.ANALYZE)
                .withGpuDecodeAttention(true)
                .buildLocalTransformerModel()) {
            assertTrue(model.tensorOperations(TensorProviderKind.GPU).isPresent(),
                    "GPU tensor operations must be available for this integration test");
            PromptContext prompt = model.promptSupport().orElseThrow().builder()
                    .addTemplateArgs(Map.of("enable_thinking", false))
                    .addUserMessage("Solve this carefully but concisely: A train leaves at 3 PM traveling 60 mph. "
                            + "Another leaves the same station at 4 PM traveling 90 mph on the same route. "
                            + "When does the second train catch the first? Explain briefly, then give the final time.")
                    .build();
            AtomicInteger k = new AtomicInteger(0);
            Response response = model.generate(UUID.randomUUID(), prompt,
                    new GeneratorParameters().withTemperature(0.7f).withTopP(0.8f).withTopK(20.0f).withMaxTokens(256),

                    new GenerateEvent() {
                        @Override
                        public void emit(int next, String nextRaw, String nextCleaned, float timing) {
                            System.out.println(k.getAndIncrement() +" "+ nextRaw+" "+System.currentTimeMillis());
                        }
                    });
            System.out.println("QWEN3_4B_JQ4_LONG_GPU_DECODE_ATTENTION="
                    + response.responseTextWithSpecialTokens.replace("\n", "\\n"));
            printTensorRuntimeCounters(model.getMetricRegistry());
            printCounter(model.getMetricRegistry(),
                    "causalselfattention.decode_paged_attention.provider_Native_GPU_Operations");
            printCounter(model.getMetricRegistry(),
                    "kvcacheselfattention.decode_paged_attention.provider_Native_GPU_Operations");
            assertFalse(response.responseTextWithSpecialTokens.isBlank());
            assertTrue(gpuDecodePagedAttentionCount(model) > 0,
                    "decode paged attention should use Native GPU Operations");
        } finally {
            InferenceProfiler.setEnabled(previousProfiling);
        }
    }

    @Disabled
    public void qwen306BThinkingPathProducesNonEmptyOutput() {
        ModelFetcher fetch = new ModelFetcher("Qwen", "Qwen3-0.6B");
        try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetch).buildLocalTransformerModel()) {
            PromptContext prompt = model.promptSupport().orElseThrow().builder()
                    .addTemplateArgs(Map.of("enable_thinking", true))
                    .addUserMessage("What is 1 + 1? Think briefly, then answer.")
                    .build();
            Response response = model.generate(UUID.randomUUID(), prompt,
                    new GeneratorParameters().withTemperature(0.6f).withTopP(0.95f).withTopK(20.0f).withMaxTokens(64),
                    new GenerateEvent() {
                        @Override
                        public void emit(int next, String nextRaw, String nextCleaned, float timing) {
                            System.out.println(nextRaw );
                        }
                    });
            System.out.println("QWEN3_06B_THINKING=" + response.responseTextWithSpecialTokens.replace("\n", "\\n"));
            assertFalse(response.responseTextWithSpecialTokens.isBlank());
            assertEquals("""
                    <think>
                    Okay, so the question is 1 plus 1. Hmm, let's think. I know that in basic arithmetic, when you add two numbers together, you just add their values. So 1 plus 1 would be 2. But wait, maybe there's something else I'm missing here?
                    """.trim(), response.responseText);
        }
    }

    @Disabled
    public void qwen306BQuantizeOnDemandThinkingPathProducesNonEmptyOutput() {
        ModelFetcher fetch = new ModelFetcher("Qwen", "Qwen3-0.6B");
        try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetch)
                .withQuantizeOnDemand(DType.Q4, "Qwen", "Qwen3-0.6B-JQ4")
                .buildLocalTransformerModel()) {
            PromptContext prompt = model.promptSupport().orElseThrow().builder()
                    .addTemplateArgs(Map.of("enable_thinking", true))
                    .addUserMessage("What is 1 + 1? Think briefly, then answer.")

                    .build();
            Response response = model.generate(UUID.randomUUID(), prompt,
                    new GeneratorParameters().withTemperature(0.6f).withTopP(0.95f).withTopK(20.0f).withMaxTokens(64),
                    new DoNothingGenerateEvent());
            System.out.println("QWEN3_06B_QOD_THINKING=" + response.responseTextWithSpecialTokens.replace("\n", "\\n"));

            assertFalse(response.responseTextWithSpecialTokens.isBlank());
            assertEquals("""
                    <think>
                    Okay, so the user is asking, "What is 1 + 1?" and wants a brief answer. Let me think. First, I need to confirm if they're asking about the mathematical operation or if there's a trick involved. In math, 1 + 1 is 2, but
                    """.trim(), response.responseText);
        }
    }

    @Disabled
    public void biggerQuantize() {
        ModelFetcher fetch = new ModelFetcher("Qwen", "Qwen3-4B");
        try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetch)
                .withQuantizeOnDemand(DType.Q4, "Qwen", "Qwen3-4B-JQ4")
                .buildLocalTransformerModel()) {
            PromptContext prompt = model.promptSupport().orElseThrow().builder()
                    .addTemplateArgs(Map.of("enable_thinking", true))
                    .addUserMessage("Think briefly, then answer in less than 3 sentences. If John has $10.00 and Ed has double what John has how much money does Edward have?")
                    .build();
            Response response = model.generate(UUID.randomUUID(), prompt,
                    new GeneratorParameters().withTemperature(0.6f).withTopP(0.95f).withTopK(20.0f).withMaxTokens(512),
                    new DoNothingGenerateEvent());
            System.out.println("QWEN3_06B_QOD_THINKING=" + response.responseTextWithSpecialTokens.replace("\n", "\\n"));
            System.out.println(response);
            assertFalse(response.responseTextWithSpecialTokens.isBlank());
            assertEquals("""
                    <think>
                    Okay, so the user is asking, "What is 1 + 1?" and wants a brief answer. Let me think. First, I need to confirm if they're asking about the mathematical operation or if there's a trick involved. In math, 1 + 1 is 2, but
                    """.trim(), response.responseText);
            //QWEN3_06B_QOD_THINKING=<think>\nOkay, let's see. John has $10.00. Ed has double that. So, double of 10 is 20. So Ed has $20.00. That's straightforward. No need for complicated steps here. Just multiply 10 by 2.\n</think>\n\nEd has $20.00, as double John's $10.00 is 2 × $10.00 = $20.00.<|im_end|>
        }
    }

    //@Disabled("Requires enough disk/RAM to quantize and run dense Qwen3-30B.")
    @Disabled("Requires enough disk/RAM to quantize and run dense Qwen3-30B.")
    //@Test
    public void qwen332BDenseQuantizeOnDemandToolCallingPromptProducesOutput() {
        ModelFetcher fetch = new ModelFetcher("Qwen", "Qwen3-32B");
        try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetch)
                .withGpuDecodeAttention(true)
                .withDownload(false)
                .withQuantizeOnDemand(DType.Q4, "Qwen", "Qwen3-32B-JQ4")
                .buildLocalTransformerModel()) {
            PromptContext prompt = model.promptSupport().orElseThrow().builder()
                    .addTemplateArgs(Map.of("enable_thinking", false))
                    .addUserMessage("List exactly three likely files in a small Java project. Answer as a plain bullet list.")
                    .build();
            Response response = model.generate(UUID.randomUUID(), prompt,
                    new GeneratorParameters().withTemperature(0.0f).withMaxTokens(96),
                    printTokens("QWEN3_30B_QOD_TOOLING"));
            System.out.println("QWEN3_30B_QOD_TOOLING=" + response.responseTextWithSpecialTokens.replace("\n", "\\n"));
            assertFalse(response.responseTextWithSpecialTokens.isBlank());
        }
    }

    private static GenerateEvent printTokens(String label) {
        return (next, nextRaw, nextCleaned, timing) -> {
            System.out.print(nextRaw == null ? "" : nextRaw);
            System.out.flush();
            System.out.printf("%n[%s token=%d elapsed=%.3fs]%n", label, next, timing);
        };
    }

    private static void printTensorRuntimeCounters(MetricRegistry metrics) {
        printCounter(metrics, "tensorruntime.affinity.pinned");
        printCounter(metrics, "tensorruntime.affinity.failed");
        printCounter(metrics, "tensorruntime.affinity.unsupported");
        printCounter(metrics, "tensorruntime.locality.local");
        printCounter(metrics, "tensorruntime.locality.remote");
        printCounter(metrics, "tensorruntime.locality.unknown");
        printCounter(metrics, "tensorruntime.tasks.submitted");
        printCounter(metrics, "tensorruntime.policy.applied");
        printCounter(metrics, "tensorruntime.policy.not_applied");
    }

    private static void printCounter(MetricRegistry metrics, String name) {
        System.out.printf("[tensor-runtime-counter] %s count=%d%n", name, metrics.counter(name).getCount());
    }

    private static long gpuDecodePagedAttentionCount(AbstractModel model) {
        return model.getMetricRegistry()
                .counter("causalselfattention.decode_paged_attention.provider_Native_GPU_Operations").getCount()
                + model.getMetricRegistry()
                .counter("kvcacheselfattention.decode_paged_attention.provider_Native_GPU_Operations").getCount();
    }
}
