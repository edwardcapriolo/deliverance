package io.teknek.deliverance.model.nemotronlabsdiffusion;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.JsonUtils;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.safetensors.DefaultWeightLoader;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.safetensors.ModelQuantizer;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorInfo;
import com.fasterxml.jackson.databind.JsonNode;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.Map;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertTrue;

@Tag("longtest")
class NemotronLabsDiffusionBaseCheckpointIT {
    private static final ModelFetcher FETCHER = new ModelFetcher("nvidia", "Nemotron-Labs-Diffusion-3B-Base");
    private static final String QOD_OWNER = "nvidia";
    private static final String QOD_MODEL = "Nemotron-Labs-Diffusion-3B-Base-JQ4";
    private static final ModelFetcher INSTRUCT_FETCHER = new ModelFetcher("nvidia", "Nemotron-Labs-Diffusion-3B");
    private static final String INSTRUCT_QOD_OWNER = "edwardcapriolo";
    private static final String INSTRUCT_QOD_MODEL = "Nemotron-Labs-Diffusion-3B-JQ4";
    private static final String BENCHMARK_MATH_PROMPT = "Solve step by step. A bus starts with an unknown number of passengers. "
            + "At the first stop, half get off and 4 get on. At the second stop, 6 get off and 8 get on. "
            + "There are 25 passengers heading to the third stop. How many passengers started at the terminal? "
            + "Then compute total fare collected if every person who ever boarded paid $2.";

    @Test
    void baseCheckpointHasExpectedTensorInventory() {
        File modelRoot = FETCHER.maybeDownload();

        try (DefaultWeightLoader loader = new DefaultWeightLoader(modelRoot)) {
            Map<String, TensorInfo> tensors = loader.tensorInfoMap();
            assertEquals(237, tensors.size());
            assertShape(tensors, "diffusion_head.weight", 131072, 3072);
            assertShape(tensors, "encoder.embed_tokens.weight", 131072, 3072);
            assertShape(tensors, "encoder.norm.weight", 3072);
            assertShape(tensors, "encoder.layers.0.input_layernorm.weight", 3072);
            assertShape(tensors, "encoder.layers.0.post_attention_layernorm.weight", 3072);
            assertShape(tensors, "encoder.layers.0.self_attn.q_proj.weight", 4096, 3072);
            assertShape(tensors, "encoder.layers.0.self_attn.k_proj.weight", 1024, 3072);
            assertShape(tensors, "encoder.layers.0.self_attn.v_proj.weight", 1024, 3072);
            assertShape(tensors, "encoder.layers.0.self_attn.o_proj.weight", 3072, 4096);
            assertShape(tensors, "encoder.layers.0.mlp.gate_proj.weight", 9216, 3072);
            assertShape(tensors, "encoder.layers.0.mlp.up_proj.weight", 9216, 3072);
            assertShape(tensors, "encoder.layers.0.mlp.down_proj.weight", 3072, 9216);
            assertShape(tensors, "encoder.layers.25.self_attn.q_proj.weight", 4096, 3072);
            assertTrue(modelRoot.toPath().resolve(".finished").toFile().isFile());
        }
    }

    @Test
    void baseCheckpointCanInstantiateNemotronModel() {
        try (AbstractModel model = AutoModelForCausaLm.newBuilder(FETCHER).buildLocalTransformerModel()) {
            assertInstanceOf(NemotronLabsDiffusionModel.class, model);
        }
    }

    @Test
    void baseCheckpointPromptTemplateRuntimeTokensDoNotPrependBos() {
        try (AbstractModel model = AutoModelForCausaLm.newBuilder(FETCHER).buildLocalTransformerModel()) {
            String prompt = model.promptSupport().orElseThrow().builder()
                    .addUserMessage("Hi!")
                    .build()
                    .getPrompt();
            int[] runtimeTokens = model.constructPromptTokensForRuntime(prompt);

            assertEquals(10, runtimeTokens[0], "Nemotron chat template should start with <|im_start|>, not BOS");
            assertTrue(runtimeTokens[0] != model.getConfig().bosToken,
                    "Nemotron tokenizer_config disables add_bos_token for chat-template prompts");
        }
    }

    @Test
    void baseCheckpointArGenerateOneTokenSmoke() {
        try (AbstractModel model = AutoModelForCausaLm.newBuilder(FETCHER).buildLocalTransformerModel()) {
            NemotronLabsDiffusionModel nemotron = (NemotronLabsDiffusionModel) model;
            Response response = nemotron.generateArBaseline(UUID.randomUUID(), PromptContext.of("The capital of France is"),
                    new GeneratorParameters().withMaxTokens(1), (next, nextRaw, nextCleaned, timing) -> { });

            assertEquals(1, response.generatedTokens.size());
            assertTrue(response.generatedTokens.getFirst() >= 0);
            assertTrue(response.generatedTokens.getFirst() < model.getConfig().vocabularySize);
        }
    }

    @Test
    void baseCheckpointArGenerateMultiTokenSmoke() {
        try (AbstractModel model = AutoModelForCausaLm.newBuilder(FETCHER).buildLocalTransformerModel()) {
            NemotronLabsDiffusionModel nemotron = (NemotronLabsDiffusionModel) model;
            Response response = nemotron.generateArBaseline(UUID.randomUUID(), PromptContext.of("The capital of France is"),
                    new GeneratorParameters().withMaxTokens(6), (next, nextRaw, nextCleaned, timing) ->
                            System.out.println("AR " + nextCleaned + " " + next));

            System.out.println("AR_TEXT=" + response.responseText);
            assertEquals(6, response.generatedTokens.size());
            assertTrue(response.responseText != null && !response.responseText.isBlank());
            assertTrue(response.responseText.toLowerCase().contains("paris"), response.responseText);
            for (int token : response.generatedTokens) {
                assertTrue(token >= 0);
                assertTrue(token < model.getConfig().vocabularySize);
            }
        }
    }

    @Test
    void baseCheckpointQuantizeOnDemandArGenerateSmoke() throws Exception {
        try (AbstractModel model = AutoModelForCausaLm.newBuilder(FETCHER)
                .withQuantizeOnDemand(DType.Q4, QOD_OWNER, QOD_MODEL)
                .buildLocalTransformerModel()) {
            assertInstanceOf(NemotronLabsDiffusionModel.class, model);
            assertQuantizedPolicy(new ModelFetcher(QOD_OWNER, QOD_MODEL).pathForModel());

            NemotronLabsDiffusionModel nemotron = (NemotronLabsDiffusionModel) model;
            Response response = nemotron.generateArBaseline(UUID.randomUUID(), PromptContext.of("The capital of France is"),
                    new GeneratorParameters().withMaxTokens(3), (next, nextRaw, nextCleaned, timing) ->
                            System.out.println("QOD_AR " + nextCleaned + " " + next));

            System.out.println("QOD_AR_TEXT=" + response.responseText);
            assertEquals(3, response.generatedTokens.size());
            assertTrue(response.responseText != null && !response.responseText.isBlank());
            assertTrue(response.responseText.toLowerCase().contains("paris"), response.responseText);
        }
    }

    @Test
    void baseCheckpointQuantizeOnDemandArGenerateOneShotProfile() throws Exception {
        boolean previousProfiling = InferenceProfiler.isEnabled();
        InferenceProfiler.setEnabled(true);
        try (AbstractModel model = AutoModelForCausaLm.newBuilder(FETCHER)
                .withQuantizeOnDemand(DType.Q4, QOD_OWNER, QOD_MODEL)
                .buildLocalTransformerModel()) {
            assertInstanceOf(NemotronLabsDiffusionModel.class, model);
            assertQuantizedPolicy(new ModelFetcher(QOD_OWNER, QOD_MODEL).pathForModel());

            NemotronLabsDiffusionModel nemotron = (NemotronLabsDiffusionModel) model;
            InferenceProfiler.reset();
            Response response = nemotron.generateArBaseline(UUID.randomUUID(), PromptContext.of("The capital of France is"),
                    new GeneratorParameters().withMaxTokens(25), (next, nextRaw, nextCleaned, timing) ->
                            System.out.println("QOD_AR_PROFILE " + nextCleaned + " " + next));

            System.out.println("QOD_AR_PROFILE_TEXT=" + response.responseText);
            InferenceProfiler.printSummary("nemotron qod ar one-shot", 40);
            printProfileCounters(model, 80);
            assertEquals(25, response.generatedTokens.size());
            assertTrue(response.responseText != null && !response.responseText.isBlank());
            assertTrue(response.responseText.toLowerCase().contains("paris"), response.responseText);
        } finally {
            InferenceProfiler.setEnabled(previousProfiling);
        }
    }


    @Test
    void baseCheckpointQuantizeOnDemandArGenerateOneShotProfileQ4Head() throws Exception {
        boolean previousProfiling = InferenceProfiler.isEnabled();
        InferenceProfiler.setEnabled(true);
        try (AbstractModel model = AutoModelForCausaLm.newBuilder(FETCHER)
                .withQuantizeOnDemand(DType.Q4, QOD_OWNER, QOD_MODEL)
                .withOutputHeadQuantization(DType.Q4)
                .buildLocalTransformerModel()) {
            assertInstanceOf(NemotronLabsDiffusionModel.class, model);

            NemotronLabsDiffusionModel nemotron = (NemotronLabsDiffusionModel) model;
            InferenceProfiler.reset();
            Response response = nemotron.generateArBaseline(UUID.randomUUID(), PromptContext.of("The capital of France is"),
                    new GeneratorParameters().withMaxTokens(25), (next, nextRaw, nextCleaned, timing) ->
                            System.out.println("QOD_AR_PROFILE_Q4_HEAD " + nextCleaned + " " + next));

            System.out.println("QOD_AR_PROFILE_Q4_HEAD_TEXT=" + response.responseText);
            InferenceProfiler.printSummary("nemotron qod ar one-shot q4 head", 40);
            printProfileCounters(model, 80);
            assertEquals(25, response.generatedTokens.size());
            assertTrue(response.responseText != null && !response.responseText.isBlank());
        } finally {
            InferenceProfiler.setEnabled(previousProfiling);
        }
    }

    @Test
    void baseCheckpointDiffusionGenerateOneTokenSmoke() {
        try (AbstractModel model = AutoModelForCausaLm.newBuilder(FETCHER).buildLocalTransformerModel()) {
            Response response = model.generate(UUID.randomUUID(), PromptContext.of("The capital of France is"),
                    new GeneratorParameters().withMaxTokens(1), (next, nextRaw, nextCleaned, timing) -> { });

            assertEquals(1, response.generatedTokens.size());
            assertTrue(response.generatedTokens.getFirst() >= 0);
            assertTrue(response.generatedTokens.getFirst() < model.getConfig().vocabularySize);
        }
    }

    @Test
    void baseCheckpointQuantizeOnDemandDiffusionBlockFourSmoke() {
        try (AbstractModel model = AutoModelForCausaLm.newBuilder(FETCHER)
                .withQuantizeOnDemand(DType.Q4, QOD_OWNER, QOD_MODEL)
                .withOutputHeadQuantization(DType.Q4)
                .buildLocalTransformerModel()) {
            Response response = model.generate(UUID.randomUUID(), PromptContext.of("Question: What is Paris?\nAnswer:"),
                    new GeneratorParameters().withMaxTokens(4).withDiffusionBlockLength(4),
                    (next, nextRaw, nextCleaned, timing) ->
                            System.out.println("DIFFUSION_BLOCK4 " + nextCleaned + " " + next));

            System.out.println("DIFFUSION_BLOCK4_TEXT=" + response.responseText);
            assertEquals(4, response.generatedTokens.size());
            for (int token : response.generatedTokens) {
                assertTrue(token >= 0);
                assertTrue(token < model.getConfig().vocabularySize);
            }
        }
    }

    @Test
    void baseCheckpointQuantizeOnDemandDiffusionBlockThirtyTwoProfile() {
        boolean previousProfiling = InferenceProfiler.isEnabled();
        InferenceProfiler.setEnabled(true);
        try (AbstractModel model = AutoModelForCausaLm.newBuilder(FETCHER)
                .withQuantizeOnDemand(DType.Q4, QOD_OWNER, QOD_MODEL)
                .withOutputHeadQuantization(DType.Q4)
                .buildLocalTransformerModel()) {
            InferenceProfiler.reset();
            Response response = model.generate(UUID.randomUUID(), PromptContext.of("Question: What is Paris?\nAnswer:"),
                    new GeneratorParameters().withMaxTokens(33),
                    (next, nextRaw, nextCleaned, timing) ->
                            System.out.println("DIFFUSION_BLOCK32 " + nextCleaned + " " + next));

            System.out.println("DIFFUSION_BLOCK32_TEXT=" + response.responseText);
            InferenceProfiler.printSummary("nemotron qod diffusion block32", 40);
            printProfileCounters(model, 80);
            assertTrue(response.generatedTokens.size() >= 2);
            assertTrue(response.generatedTokens.size() <= 33);
            assertTrue(response.responseText != null && !response.responseText.isBlank());
        } finally {
            InferenceProfiler.setEnabled(previousProfiling);
        }
    }

    @Test
    void baseCheckpointQuantizeOnDemandDiffusionBlockThirtyTwoProfileGpuBlockLogits() {
        boolean previousProfiling = InferenceProfiler.isEnabled();
        InferenceProfiler.setEnabled(true);
        try (AbstractModel model = AutoModelForCausaLm.newBuilder(FETCHER)
                .withQuantizeOnDemand(DType.Q4, QOD_OWNER, QOD_MODEL)
                .withOutputHeadQuantization(DType.Q4)
                .withGpuDiffusionBlockProjection(true)
                .withPackedBlockAttention(true)
                .buildLocalTransformerModel()) {
            InferenceProfiler.reset();


            for (int i=0;i< 10; i++) {
                //Solve step by step. A bus starts with an unknown number of passengers. At the first stop, half get off and 4 get on. At the second stop, 6 get off and 8 get on. There are 25 passengers heading to the third stop. How many passengers started at the terminal? Then compute total fare collected if every person who ever boarded paid $2.
                //PromptContext pc = model.promptSupport().get().builder().addUserMessage("What is Paris?").build();
                PromptContext pc = model.promptSupport().get().builder().addUserMessage("Solve step by step. A bus starts with an unknown number of passengers. At the first stop, half get off and 4 get on. At the second stop, 6 get off and 8 get on. There are 25 passengers heading to the third stop. How many passengers started at the terminal? Then compute total fare collected if every person who ever boarded paid $2.").build();
                Response response = model.generate(UUID.randomUUID(), pc,

            //for (int i=0;i< 10; i++) {
            //    PromptContext pc = model.promptSupport().get().builder().addUserMessage("What is Paris?").build();
            //    Response response = model.generate(UUID.randomUUID(), PromptContext.of("Question: What is Paris?\nAnswer:"),
                        new GeneratorParameters().withMaxTokens(33),
                        (next, nextRaw, nextCleaned, timing) ->
                                System.out.println("DIFFUSION_BLOCK32_GPU_LOGITS " + nextCleaned + " " + next));

                System.out.println("DIFFUSION_BLOCK32_GPU_LOGITS_TEXT=" + response.responseText);
                InferenceProfiler.printSummary("nemotron qod diffusion block32 gpu block logits", 40);
                printProfileCounters(model, 80);
                assertTrue(response.generatedTokens.size() >= 2);
                assertTrue(response.generatedTokens.size() <= 33);
                assertTrue(response.responseText != null && !response.responseText.isBlank());
                assertTrue(InferenceProfiler.counterValue(
                        "nemotron_labs_diffusion.logits_block_projection.provider_gpu") > 0);
            }
        } finally {
            InferenceProfiler.setEnabled(previousProfiling);
        }
    }

    @Test
    void instructCheckpointQuantizeOnDemandDiffusionBenchmarkMathPromptProfileGpuBlockLogits() {
        boolean previousProfiling = InferenceProfiler.isEnabled();
        InferenceProfiler.setEnabled(true);
        try (AbstractModel model = AutoModelForCausaLm.newBuilder(INSTRUCT_FETCHER)
                .withQuantizeOnDemand(DType.Q4, INSTRUCT_QOD_OWNER, INSTRUCT_QOD_MODEL)
                .withOutputHeadQuantization(DType.Q4)
                .withGpuDiffusionBlockProjection(true)
                .withPackedBlockAttention(true)
                .withKvBlockStoragePolicy(KvBufferCacheSettings.KvBlockStoragePolicy.MSE_TURBOQUANT)
                .withKvTurboQuantBits(4)
                .withTrackKvReadViews(true)
                .buildLocalTransformerModel()) {
            PromptContext prompt = model.promptSupport().orElseThrow().builder()
                    .addUserMessage(BENCHMARK_MATH_PROMPT)
                    .build();
            InferenceProfiler.reset();
            Response response = model.generate(UUID.randomUUID(), prompt,
                    new GeneratorParameters().withMaxTokens(128),
                    (next, nextRaw, nextCleaned, timing) ->
                            System.out.println("INSTRUCT_DIFFUSION_MATH " + nextCleaned + " " + next));

            System.out.println("INSTRUCT_DIFFUSION_MATH_TEXT=" + response.responseText);
            System.out.println("INSTRUCT_DIFFUSION_MATH_RESPONSE=" + response);
            InferenceProfiler.printSummary("nemotron instruct qod diffusion math prompt gpu block logits", 40);
            printProfileCounters(model, 80);
            assertTrue(response.responseText != null && !response.responseText.isBlank());
        } finally {
            InferenceProfiler.setEnabled(previousProfiling);
        }
    }

    @Test
    void instructCheckpointQuantizeOnDemandDiffusionBenchmarkMathPromptProfileCpuDenseKv() {
        boolean previousProfiling = InferenceProfiler.isEnabled();
        InferenceProfiler.setEnabled(true);
        try (AbstractModel model = AutoModelForCausaLm.newBuilder(INSTRUCT_FETCHER)
                .withQuantizeOnDemand(DType.Q4, INSTRUCT_QOD_OWNER, INSTRUCT_QOD_MODEL)
                .withOutputHeadQuantization(DType.Q4)
                .withPackedBlockAttention(true)
                .buildLocalTransformerModel()) {
            PromptContext prompt = model.promptSupport().orElseThrow().builder()
                    .addUserMessage(BENCHMARK_MATH_PROMPT)
                    .build();
            InferenceProfiler.reset();
            Response response = model.generate(UUID.randomUUID(), prompt,
                    new GeneratorParameters().withMaxTokens(32),
                    (next, nextRaw, nextCleaned, timing) ->
                            System.out.println("INSTRUCT_DIFFUSION_MATH_CPU_DENSE " + nextCleaned + " " + next));

            System.out.println("INSTRUCT_DIFFUSION_MATH_CPU_DENSE_TEXT=" + response.responseText);
            System.out.println("INSTRUCT_DIFFUSION_MATH_CPU_DENSE_RESPONSE=" + response);
            InferenceProfiler.printSummary("nemotron instruct qod diffusion math prompt cpu dense kv", 40);
            printProfileCounters(model, 100);
            assertTrue(response.responseText != null && !response.responseText.isBlank());
        } finally {
            InferenceProfiler.setEnabled(previousProfiling);
        }
    }

    @Test
    void instructCheckpointQuantizeOnDemandDiffusionBenchmarkMathPromptProfileCpuTurboQuantKv() {
        boolean previousProfiling = InferenceProfiler.isEnabled();
        InferenceProfiler.setEnabled(true);
        try (AbstractModel model = AutoModelForCausaLm.newBuilder(INSTRUCT_FETCHER)
                .withQuantizeOnDemand(DType.Q4, INSTRUCT_QOD_OWNER, INSTRUCT_QOD_MODEL)
                .withOutputHeadQuantization(DType.Q4)
                .withPackedBlockAttention(true)
                .withKvBlockStoragePolicy(KvBufferCacheSettings.KvBlockStoragePolicy.MSE_TURBOQUANT)
                .withKvTurboQuantBits(4)
                .buildLocalTransformerModel()) {
            PromptContext prompt = model.promptSupport().orElseThrow().builder()
                    .addUserMessage(BENCHMARK_MATH_PROMPT)
                    .build();
            InferenceProfiler.reset();
            Response response = model.generate(UUID.randomUUID(), prompt,
                    new GeneratorParameters().withMaxTokens(32),
                    (next, nextRaw, nextCleaned, timing) ->
                            System.out.println("INSTRUCT_DIFFUSION_MATH_CPU_TURBOQUANT " + nextCleaned + " " + next));

            System.out.println("INSTRUCT_DIFFUSION_MATH_CPU_TURBOQUANT_TEXT=" + response.responseText);
            System.out.println("INSTRUCT_DIFFUSION_MATH_CPU_TURBOQUANT_RESPONSE=" + response);
            InferenceProfiler.printSummary("nemotron instruct qod diffusion math prompt cpu turboquant kv", 40);
            printProfileCounters(model, 100);
            assertTrue(response.responseText != null && !response.responseText.isBlank());
        } finally {
            InferenceProfiler.setEnabled(previousProfiling);
        }
    }

    @Test
    void baseCheckpointQuantizeOnDemandDiffusionBlockThirtyTwoProfileWithLinearSpecLora() {
        boolean previousProfiling = InferenceProfiler.isEnabled();
        InferenceProfiler.setEnabled(true);
        try (AbstractModel model = AutoModelForCausaLm.newBuilder(FETCHER)
                .withQuantizeOnDemand(DType.Q4, QOD_OWNER, QOD_MODEL)
                .withOutputHeadQuantization(DType.Q4)
                .buildLocalTransformerModel()) {
            NemotronLabsDiffusionModel nemotron = (NemotronLabsDiffusionModel) model;
            nemotron.registerLinearSpecLoraAdapter();
            InferenceProfiler.reset();
            Response response = model.generate(UUID.randomUUID(), PromptContext.of("Question: What is Paris?\nAnswer:"),
                    new GeneratorParameters().withMaxTokens(33),
                    (next, nextRaw, nextCleaned, timing) ->
                            System.out.println("DIFFUSION_BLOCK32_LORA " + nextCleaned + " " + next));

            System.out.println("DIFFUSION_BLOCK32_LORA_TEXT=" + response.responseText);
            InferenceProfiler.printSummary("nemotron qod diffusion block32 linear-spec-lora", 40);
            printProfileCounters(model, 80);
            assertTrue(response.generatedTokens.size() >= 2);
            assertTrue(response.generatedTokens.size() <= 33);
            assertTrue(response.responseText != null && !response.responseText.isBlank());
        } finally {
            InferenceProfiler.setEnabled(previousProfiling);
        }
    }


    @Test
    void baseCheckpointDiffusionMultiTokenSmoke() {
        try (AbstractModel model = AutoModelForCausaLm.newBuilder(new ModelFetcher(QOD_OWNER, QOD_MODEL)).buildLocalTransformerModel()) {
            Response response = model.generate(UUID.randomUUID(), PromptContext.of("Give me a history of Lake George"),
                    new GeneratorParameters().withMaxTokens(1024), (next, nextRaw, nextCleaned, timing) -> {
                System.out.println(nextCleaned + " " + next);
                    });

            System.out.println(response);
            assertEquals(2, response.generatedTokens.size());
            for (int token : response.generatedTokens) {
                assertTrue(token >= 0);
                assertTrue(token < model.getConfig().vocabularySize);
            }
        }
    }

    private static void assertShape(Map<String, TensorInfo> tensors, String name, int... shape) {
        TensorInfo info = tensors.get(name);
        assertTrue(info != null, "missing tensor " + name);
        assertArrayEquals(shape, info.shape, name);
    }

    private static void assertQuantizedPolicy(Path modelRoot) throws Exception {
        try (DefaultWeightLoader loader = new DefaultWeightLoader(modelRoot.toFile())) {
            Map<String, TensorInfo> tensors = loader.tensorInfoMap();
            assertEquals(DType.Q4, tensors.get("encoder.layers.0.self_attn.q_proj.weight").dType);
            assertEquals(DType.F32, tensors.get("encoder.layers.0.self_attn.q_proj.weight.qb").dType);
            assertEquals(DType.Q4, tensors.get("encoder.layers.0.mlp.down_proj.weight").dType);
            assertEquals(DType.F32, tensors.get("encoder.layers.0.mlp.down_proj.weight.qb").dType);
            assertEquals(DType.BF16, tensors.get("encoder.embed_tokens.weight").dType);
            assertEquals(DType.BF16, tensors.get("encoder.norm.weight").dType);
            assertEquals(DType.BF16, tensors.get("encoder.layers.0.input_layernorm.weight").dType);
            assertEquals(DType.BF16, tensors.get("encoder.layers.0.post_attention_layernorm.weight").dType);
            assertEquals(DType.BF16, tensors.get("diffusion_head.weight").dType);
        }
        JsonNode manifest = JsonUtils.om.readTree(modelRoot.resolve(ModelQuantizer.QUANTIZATION_MANIFEST).toFile());
        assertEquals("Q4", manifest.get("targetType").asText());
        assertTrue(manifest.get("tensorTransforms").isArray());
    }

    private static void printProfileCounters(AbstractModel model, int maxRows) {
        model.getMetricRegistry().getCounters().entrySet().stream()
                .filter(entry -> entry.getValue().getCount() != 0
                        || InferenceProfiler.shouldPrintCounter(entry.getKey()))
                .sorted(Comparator.comparingLong(
                        (Map.Entry<io.dropwizard.metrics5.MetricName, io.dropwizard.metrics5.Counter> entry) ->
                                Math.abs(entry.getValue().getCount())).reversed())
                .limit(maxRows)
                .forEach(entry -> System.out.println("[profile-counter] " + InferenceProfiler.displayName(entry.getKey())
                        + " count=" + entry.getValue().getCount()
                        + " delta=" + InferenceProfiler.counterValue(entry.getKey())));
    }
}
