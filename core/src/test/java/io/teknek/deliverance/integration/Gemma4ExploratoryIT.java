package io.teknek.deliverance.integration;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.LayerNorm;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.generator.SampleOutput;
import io.teknek.deliverance.grace.TokenIds;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.model.DoNothingGenerateEvent;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.KvBufferCache;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Assumptions;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.lang.reflect.Field;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.EnumMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.UUID;
import java.util.stream.IntStream;

@Tag("large-model")
public class Gemma4ExploratoryIT {
    private static final ObjectMapper OM = new ObjectMapper();

    @Test
    public void checkpointQuantizationLayout() throws Exception {
        File modelRoot = new ModelFetcher("edward", "gemma-4-E2B-it-JQ4").maybeDownload();
        EnumMap<DType, Integer> counts = new EnumMap<>(DType.class);
        List<String> q4Samples = new ArrayList<>();
        for (File file : modelRoot.listFiles((dir, name) -> name.endsWith(".safetensors"))) {
            JsonNode header = safetensorsHeader(file);
            header.fields().forEachRemaining(entry -> {
                if (entry.getKey().equals("__metadata__")) {
                    return;
                }
                String dtype = entry.getValue().path("dtype").asText();
                DType deliveranceType = switch (dtype) {
                    case "F32" -> DType.F32;
                    case "BF16" -> DType.BF16;
                    case "F16" -> DType.F16;
                    case "Q4" -> DType.Q4;
                    case "I8" -> DType.I8;
                    default -> null;
                };
                if (deliveranceType != null && !entry.getKey().endsWith(".qb")) {
                    counts.put(deliveranceType, counts.getOrDefault(deliveranceType, 0) + 1);
                    if (deliveranceType == DType.Q4 && q4Samples.size() < 20) {
                        q4Samples.add(entry.getKey());
                    }
                }
            });
        }
        System.out.println("GEMMA4_LAYOUT_COUNTS=" + counts);
        System.out.println("GEMMA4_Q4_SAMPLES=" + q4Samples);
    }

    @Test
    public void promptThinkingVariantsAndFirstTokenTopK() throws Exception {
        AbstractModel model = Gemma4ExploratorySuite.getOrCreate();
        printPromptAndTopK(model, false);
        printPromptAndTopK(model, true);
    }

    @Test
    public void defaultGenerationShortOutput() {
        AbstractModel model = Gemma4ExploratorySuite.getOrCreate();
        PromptContext prompt = prompt(model, false);
        Response response = model.generate(UUID.randomUUID(), prompt,
                new GeneratorParameters().withTemperature(0.0f).withMaxTokens(32), new DoNothingGenerateEvent());
        System.out.println("GEMMA4_DEFAULT_SHORT_OUTPUT=" + response.responseTextWithSpecialTokens);
    }

    @Test
    public void q4SamplingShortOutputs() {
        AbstractModel model = Gemma4ExploratorySuite.getOrCreate();
        PromptContext prompt = prompt(model, false);
        Response tempOnly = model.generate(UUID.randomUUID(), prompt,
                new GeneratorParameters().withTemperature(1.0f).withMaxTokens(8).withSeed(99999), new DoNothingGenerateEvent());
        Response topP = model.generate(UUID.randomUUID(), prompt,
                new GeneratorParameters().withTemperature(1.0f).withTopP(0.95f).withMaxTokens(8).withSeed(99999), new DoNothingGenerateEvent());
        Response topK = model.generate(UUID.randomUUID(), prompt,
                new GeneratorParameters().withTemperature(1.0f).withTopK(64f).withMaxTokens(8).withSeed(99999), new DoNothingGenerateEvent());
        System.out.println("GEMMA4_Q4_TEMP1_SHORT_OUTPUT=" + tempOnly.responseTextWithSpecialTokens);
        System.out.println("GEMMA4_Q4_TEMP1_TOPP095_SHORT_OUTPUT=" + topP.responseTextWithSpecialTokens);
        System.out.println("GEMMA4_Q4_TEMP1_TOPK64_SHORT_OUTPUT=" + topK.responseTextWithSpecialTokens);
    }

    @Test
    public void thinkingGenerationShortOutput() {
        AbstractModel model = Gemma4ExploratorySuite.getOrCreate();
        PromptContext prompt = prompt(model, true);
        Response response = model.generate(UUID.randomUUID(), prompt,
                new GeneratorParameters().withTemperature(0.0f).withMaxTokens(32), new DoNothingGenerateEvent());
        System.out.println("GEMMA4_THINKING_SHORT_OUTPUT=" + response.responseTextWithSpecialTokens);
    }

    @Test
    public void thinkingThreeTokenVariants() {
        AbstractModel model = Gemma4ExploratorySuite.getOrCreate();
        String rendered = prompt(model, true).getPrompt();
        generateThree(model, "template", PromptContext.of(rendered));
        generateThree(model, "seed_channel", PromptContext.of(rendered + "<|channel>"));
        generateThree(model, "seed_thought_channel", PromptContext.of(rendered + "<|channel>thought\n"));
        generateThree(model, "seed_empty_thought", PromptContext.of(rendered + "<|channel>thought\n<channel|>"));
    }

    @Test
    public void thinkingRecommendedSamplingThreeTokens() {
        AbstractModel model = Gemma4ExploratorySuite.getOrCreate();
        Response response = model.generate(UUID.randomUUID(), prompt(model, true),
                new GeneratorParameters()
                        .withTemperature(1.0f)
                        .withTopP(0.95f)
                        .withTopK(64.0f)
                        .withMaxTokens(3)
                        .withSeed(99999),
                new DoNothingGenerateEvent());
        System.out.println("GEMMA4_THINK3_RECOMMENDED=" + response.responseTextWithSpecialTokens.replace("\n", "\\n"));
    }

    @Test
    public void thinkingRecommendedSamplingShortOutput() {
        AbstractModel model = Gemma4ExploratorySuite.getOrCreate();
        Response response = model.generate(UUID.randomUUID(), prompt(model, true),
                new GeneratorParameters()
                        .withTemperature(1.0f)
                        .withTopP(0.95f)
                        .withTopK(64.0f)
                        .withMaxTokens(16)
                        .withSeed(99999),
                new DoNothingGenerateEvent());
        System.out.println("GEMMA4_THINK16_RECOMMENDED=" + response.responseTextWithSpecialTokens.replace("\n", "\\n"));
    }

    @Test
    public void simpleReasoningRecommendedSamplingShortOutput() {
        AbstractModel model = Gemma4ExploratorySuite.getOrCreate();
        PromptContext prompt = model.promptSupport().orElseThrow().builder()
                .addTemplateArgs(Map.of("enable_thinking", true))
                .addUserMessage("Bob is a carpenter. Sara is a teacher. Who should you call to fix your roof? Answer with one name.")
                .build();
        Response response = model.generate(UUID.randomUUID(), prompt,
                new GeneratorParameters()
                        .withTemperature(1.0f)
                        .withTopP(0.95f)
                        .withTopK(64.0f)
                        .withMaxTokens(32)
                        .withSeed(99999),
                new DoNothingGenerateEvent());
        System.out.println("GEMMA4_SIMPLE_REASONING_RECOMMENDED=" + response.responseTextWithSpecialTokens.replace("\n", "\\n"));
    }

    @Test
    public void trivialReasoningRecommendedSamplingShortOutput() {
        AbstractModel model = Gemma4ExploratorySuite.getOrCreate();
        PromptContext prompt = model.promptSupport().orElseThrow().builder()
                .addTemplateArgs(Map.of("enable_thinking", true))
                .addUserMessage("What is 1 + 1? Answer with the number only.")
                .build();
        Response response = model.generate(UUID.randomUUID(), prompt,
                new GeneratorParameters()
                        .withTemperature(1.0f)
                        .withTopP(0.95f)
                        .withTopK(64.0f)
                        .withMaxTokens(24)
                        .withSeed(99999),
                new DoNothingGenerateEvent());
        System.out.println("GEMMA4_TRIVIAL_REASONING_RECOMMENDED=" + response.responseTextWithSpecialTokens.replace("\n", "\\n"));
    }

    @Test
    public void trivialReasoningFirstTokenTopK() throws Exception {
        AbstractModel model = Gemma4ExploratorySuite.getOrCreate();
        PromptContext prompt = model.promptSupport().orElseThrow().builder()
                .addTemplateArgs(Map.of("enable_thinking", true))
                .addUserMessage("What is 1 + 1? Answer with the number only.")
                .build();
        int[] finalIds = model.constructPromptTokensForRuntime(prompt.getPrompt());
        System.out.println("GEMMA4_TRIVIAL_PROMPT=" + prompt.getPrompt().replace("\n", "\\n"));
        System.out.println("GEMMA4_TRIVIAL_IDS=" + java.util.Arrays.toString(finalIds));
        try (KvBufferCache.KvBuffer kv = model.newKvBuffer();
             AbstractTensor hidden = model.batchForward(finalIds, 0, kv);
             AbstractTensor logits = logits(model, hidden)) {
            System.out.println("GEMMA4_TRIVIAL_FIRST_TOP20=" + topK(model, logits, 20));
        }
    }

    @Test
    public void trivialReasoningNoFormatInstructionFirstTokenTopK() throws Exception {
        AbstractModel model = Gemma4ExploratorySuite.getOrCreate();
        PromptContext prompt = model.promptSupport().orElseThrow().builder()
                .addTemplateArgs(Map.of("enable_thinking", true))
                .addUserMessage("What is 1 + 1?")
                .build();
        int[] finalIds = model.constructPromptTokensForRuntime(prompt.getPrompt());
        System.out.println("GEMMA4_TRIVIAL_NO_FORMAT_PROMPT=" + prompt.getPrompt().replace("\n", "\\n"));
        try (KvBufferCache.KvBuffer kv = model.newKvBuffer();
             AbstractTensor hidden = model.batchForward(finalIds, 0, kv);
             AbstractTensor logits = logits(model, hidden)) {
            System.out.println("GEMMA4_TRIVIAL_NO_FORMAT_FIRST_TOP20=" + topK(model, logits, 20));
        }
    }

    @Test
    public void trivialReasoningNoFormatInstructionRecommendedSampling() {
        AbstractModel model = Gemma4ExploratorySuite.getOrCreate();
        PromptContext prompt = model.promptSupport().orElseThrow().builder()
                .addTemplateArgs(Map.of("enable_thinking", true))
                .addUserMessage("What is 1 + 1?")
                .build();
        Response response = model.generate(UUID.randomUUID(), prompt,
                new GeneratorParameters()
                        .withTemperature(1.0f)
                        .withTopP(0.95f)
                        .withTopK(64.0f)
                        .withMaxTokens(24)
                        .withSeed(99999),
                new DoNothingGenerateEvent());
        System.out.println("GEMMA4_TRIVIAL_NO_FORMAT_RECOMMENDED=" + response.responseTextWithSpecialTokens.replace("\n", "\\n"));
    }

    @Test
    public void trivialReasoningRecommendedSamplingSeedSweep() {
        AbstractModel model = Gemma4ExploratorySuite.getOrCreate();
        PromptContext prompt = model.promptSupport().orElseThrow().builder()
                .addTemplateArgs(Map.of("enable_thinking", true))
                .addUserMessage("What is 1 + 1?")
                .build();
        for (int seed : List.of(1, 2, 3, 4, 5)) {
            Response response = model.generate(UUID.randomUUID(), prompt,
                    new GeneratorParameters()
                            .withTemperature(1.0f)
                            .withTopP(0.95f)
                            .withTopK(64.0f)
                            .withMaxTokens(12)
                            .withSeed(seed),
                    new DoNothingGenerateEvent());
            System.out.println("GEMMA4_TRIVIAL_SEED_" + seed + "=" + response.responseTextWithSpecialTokens.replace("\n", "\\n"));
        }
    }

    @Test
    public void trivialReasoningNoFormatInstructionTemperatureZero() {
        AbstractModel model = Gemma4ExploratorySuite.getOrCreate();
        PromptContext prompt = model.promptSupport().orElseThrow().builder()
                .addTemplateArgs(Map.of("enable_thinking", true))
                .addUserMessage("What is 1 + 1?")
                .build();
        Response response = model.generate(UUID.randomUUID(), prompt,
                new GeneratorParameters().withTemperature(0.0f).withMaxTokens(24), new DoNothingGenerateEvent());
        System.out.println("GEMMA4_TRIVIAL_NO_FORMAT_TEMP0=" + response.responseTextWithSpecialTokens.replace("\n", "\\n"));
    }

    @Test
    public void trivialNoThinkingTemperatureZero() {
        AbstractModel model = Gemma4ExploratorySuite.getOrCreate();
        PromptContext prompt = model.promptSupport().orElseThrow().builder()
                .addTemplateArgs(Map.of("enable_thinking", false))
                .addUserMessage("What is 1 + 1?")
                .build();
        Response response = model.generate(UUID.randomUUID(), prompt,
                new GeneratorParameters().withTemperature(0.0f).withMaxTokens(12), new DoNothingGenerateEvent());
        System.out.println("GEMMA4_TRIVIAL_NO_THINK_TEMP0=" + response.responseTextWithSpecialTokens.replace("\n", "\\n"));
    }

    @Test
    public void trivialThinkingWithSystemMessageTemperatureZero() {
        AbstractModel model = Gemma4ExploratorySuite.getOrCreate();
        PromptContext prompt = model.promptSupport().orElseThrow().builder()
                .addTemplateArgs(Map.of("enable_thinking", true))
                .addSystemMessage("You are a helpful assistant.")
                .addUserMessage("What is 1 + 1?")
                .build();
        Response response = model.generate(UUID.randomUUID(), prompt,
                new GeneratorParameters().withTemperature(0.0f).withMaxTokens(24), new DoNothingGenerateEvent());
        System.out.println("GEMMA4_TRIVIAL_THINK_SYSTEM_TEMP0=" + response.responseTextWithSpecialTokens.replace("\n", "\\n"));
    }

    @Test
    public void trivialThinkingGeneratedChannelThenForcedThoughtTopK() throws Exception {
        AbstractModel model = Gemma4ExploratorySuite.getOrCreate();
        PromptContext basePrompt = model.promptSupport().orElseThrow().builder()
                .addTemplateArgs(Map.of("enable_thinking", true))
                .addUserMessage("What is 1 + 1?")
                .build();
        String unseeded = basePrompt.getPrompt().replace("<|channel>thought\n", "");
        int[] promptTokens = model.constructPromptTokensForRuntime(unseeded);
        int channel = model.getTokenizer().encode("<|channel>").inputIds()[0];
        int thought = model.getTokenizer().encode("thought").inputIds()[0];
        int newline = model.getTokenizer().encode("\n").inputIds()[0];
        try (KvBufferCache.KvBuffer kv = model.newKvBuffer();
             AbstractTensor promptOutput = model.batchForward(promptTokens, 0, kv);
             AbstractTensor channelOutput = model.forward(channel, promptTokens.length, kv);
             AbstractTensor thoughtOutput = model.forward(thought, promptTokens.length + 1, kv);
             AbstractTensor newlineOutput = model.forward(newline, promptTokens.length + 2, kv);
             AbstractTensor logits = logits(model, newlineOutput)) {
            promptOutput.close();
            channelOutput.close();
            thoughtOutput.close();
            System.out.println("GEMMA4_FORCED_THOUGHT_TOKENS channel=" + channel + " thought=" + thought + " newline=" + newline);
            System.out.println("GEMMA4_FORCED_THOUGHT_TOP20=" + topK(model, logits, 20));
        }
    }

    @Test
    public void trivialThinkingChannelLabelVariantsTopK() throws Exception {
        AbstractModel model = Gemma4ExploratorySuite.getOrCreate();
        PromptContext basePrompt = model.promptSupport().orElseThrow().builder()
                .addTemplateArgs(Map.of("enable_thinking", true))
                .addUserMessage("What is 1 + 1?")
                .build();
        String unseeded = basePrompt.getPrompt().replace("<|channel>thought\n", "");
        printTopKForRawPrompt(model, "label_thought", unseeded + "<|channel>thought\n");
        printTopKForRawPrompt(model, "label_space_thought", unseeded + "<|channel> thought\n");
        printTopKForRawPrompt(model, "label_thought_space", unseeded + "<|channel>thought \n");
    }

    @Test
    public void trivialThinkingControlTransitionTopK() throws Exception {
        AbstractModel model = Gemma4ExploratorySuite.getOrCreate();
        PromptContext basePrompt = model.promptSupport().orElseThrow().builder()
                .addTemplateArgs(Map.of("enable_thinking", true))
                .addUserMessage("What is 1 + 1?")
                .build();
        String unseeded = basePrompt.getPrompt().replace("<|channel>thought\n", "");
        printTopKForRawPrompt(model, "after_model_turn", unseeded);
        printTopKForRawPrompt(model, "after_channel", unseeded + "<|channel>");
        printTopKForRawPrompt(model, "after_thought_label", unseeded + "<|channel>thought\n");
        printTopKForRawPrompt(model, "after_closed_thought", unseeded + "<|channel>thought\n<channel|>");
    }

    @Test
    public void trivialThinkingForcedClosedThoughtGeneration() {
        AbstractModel model = Gemma4ExploratorySuite.getOrCreate();
        PromptContext basePrompt = model.promptSupport().orElseThrow().builder()
                .addTemplateArgs(Map.of("enable_thinking", true))
                .addUserMessage("What is 1 + 1?")
                .build();
        String forcedClosed = basePrompt.getPrompt().replace("<|channel>thought\n", "<|channel>thought\n<channel|>");
        Response response = model.generate(UUID.randomUUID(), PromptContext.of(forcedClosed),
                new GeneratorParameters().withTemperature(0.0f).withMaxTokens(12), new DoNothingGenerateEvent());
        System.out.println("GEMMA4_FORCED_CLOSED_THOUGHT_TEMP0=" + response.responseTextWithSpecialTokens.replace("\n", "\\n"));
    }

    @Test
    public void trivialThinkingSpaceThoughtLabelGeneration() {
        AbstractModel model = Gemma4ExploratorySuite.getOrCreate();
        PromptContext basePrompt = model.promptSupport().orElseThrow().builder()
                .addTemplateArgs(Map.of("enable_thinking", true))
                .addUserMessage("What is 1 + 1?")
                .build();
        String spacedThoughtPrompt = basePrompt.getPrompt().replace("<|channel>thought\n", "<|channel> thought\n");
        Response response = model.generate(UUID.randomUUID(), PromptContext.of(spacedThoughtPrompt),
                new GeneratorParameters().withTemperature(0.0f).withMaxTokens(16), new DoNothingGenerateEvent());
        System.out.println("GEMMA4_SPACE_THOUGHT_TEMP0=" + response.responseTextWithSpecialTokens.replace("\n", "\\n"));
    }

    @Test
    public void trivialReasoningDecodeLogitsVsColdReplay() throws Exception {
        AbstractModel model = Gemma4ExploratorySuite.getOrCreate();
        PromptContext prompt = model.promptSupport().orElseThrow().builder()
                .addTemplateArgs(Map.of("enable_thinking", true))
                .addUserMessage("What is 1 + 1?")
                .build();
        Response response = model.generate(UUID.randomUUID(), prompt,
                new GeneratorParameters().withTemperature(0.0f).withMaxTokens(6), new DoNothingGenerateEvent());
        int[] promptTokens = model.constructPromptTokensForRuntime(prompt.getPrompt());
        int[] generated = response.generatedTokens.stream().mapToInt(Integer::intValue).toArray();
        System.out.println("GEMMA4_TRIVIAL_TEMP0_TOKENS=" + java.util.Arrays.toString(generated));
        try (KvBufferCache.KvBuffer decodeKv = model.newKvBuffer();
             AbstractTensor promptOutput = model.batchForward(promptTokens, 0, decodeKv)) {
            promptOutput.close();
            for (int i = 0; i < Math.min(5, generated.length - 1); i++) {
                int previousToken = generated[i];
                int position = promptTokens.length + i;
                try (AbstractTensor decodeOutput = model.forward(previousToken, position, decodeKv);
                     AbstractTensor replayOutput = coldReplay(model, promptTokens, generated, i + 1);
                     AbstractTensor decodeLogits = logits(model, decodeOutput);
                     AbstractTensor replayLogits = logits(model, replayOutput)) {
                    System.out.println("GEMMA4_TRIVIAL_DECODE_REPLAY step=" + i
                            + " previous=" + previousToken
                            + " decodeTop=" + topK(model, decodeLogits, 5)
                            + " replayTop=" + topK(model, replayLogits, 5)
                            + " drift=" + driftLastBatchRow(replayOutput, decodeOutput));
                }
            }
        }
    }

    @Test
    public void panamaF32GenerationShortOutput() {
        ModelFetcher fetch = new ModelFetcher("edward", "gemma-4-E2B-it-JQ4");
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores());
             AbstractModel model = AutoModelForCausaLm.newBuilder(fetch)
                     .withWorkingQuantType(DType.F32)
                     .withTensorProvider(new ConfigurableTensorProvider(new ArrayQueueTensorAllocator(new com.codahale.metrics.MetricRegistry()), pool))
                     .buildLocalTransformerModel()) {
            PromptContext prompt = prompt(model, false);
            Response response = model.generate(UUID.randomUUID(), prompt,
                    new GeneratorParameters().withTemperature(0.0f).withMaxTokens(32), new DoNothingGenerateEvent());
            System.out.println("GEMMA4_PANAMA_F32_SHORT_OUTPUT=" + response.responseTextWithSpecialTokens);
        }
    }

    @Test
    public void denseGooglePanamaF32GenerationShortOutput() {
        Assumptions.assumeTrue(Boolean.getBoolean("deliverance.gemma4.explore.dense"),
                "Set -Ddeliverance.gemma4.explore.dense=true to load dense google/gemma-4-E2B-it");
        File denseDir = new ModelFetcher("google", "gemma-4-E2B-it").pathForModel().toFile();
        Assumptions.assumeTrue(denseDir.isDirectory(), "Dense google/gemma-4-E2B-it cache is not present");
        ModelFetcher fetch = new ModelFetcher("google", "gemma-4-E2B-it");
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores());
             AbstractModel model = AutoModelForCausaLm.newBuilder(fetch)
                     .withWorkingQuantType(DType.F32)
                     .withTensorProvider(new ConfigurableTensorProvider(new ArrayQueueTensorAllocator(new com.codahale.metrics.MetricRegistry()), pool))
                     .buildLocalTransformerModel()) {
            PromptContext prompt = prompt(model, false);
            Response response = model.generate(UUID.randomUUID(), prompt,
                    new GeneratorParameters().withTemperature(0.0f).withMaxTokens(32), new DoNothingGenerateEvent());
            System.out.println("GEMMA4_DENSE_GOOGLE_PANAMA_F32_SHORT_OUTPUT=" + response.responseTextWithSpecialTokens);
        }
    }

    @Test
    public void batchPrefillTokenByTokenAndDecodeColdReplayDrift() {
        AbstractModel model = Gemma4ExploratorySuite.getOrCreate();
        int[] promptTokens = model.constructPromptTokensForRuntime(prompt(model, false).getPrompt());
        int[] continuation = IntStream.of(toInts(model.encodeForRuntime(" This is a classic overthought scenario question.")))
                .limit(4)
                .toArray();
        try (KvBufferCache.KvBuffer batchKv = model.newKvBuffer();
             KvBufferCache.KvBuffer stepKv = model.newKvBuffer();
             AbstractTensor batchOutput = model.batchForward(promptTokens, 0, batchKv);
             AbstractTensor stepOutput = tokenByTokenPrefill(model, promptTokens, stepKv)) {
            System.out.println("GEMMA4_PREFILL_DRIFT=" + driftLastBatchRow(batchOutput, stepOutput));
        }
        try (KvBufferCache.KvBuffer decodeKv = model.newKvBuffer();
             AbstractTensor promptOutput = model.batchForward(promptTokens, 0, decodeKv)) {
            promptOutput.close();
            for (int i = 0; i < continuation.length; i++) {
                int position = promptTokens.length + i;
                try (AbstractTensor decodeOutput = model.forward(continuation[i], position, decodeKv);
                     AbstractTensor replayOutput = coldReplay(model, promptTokens, continuation, i + 1)) {
                    System.out.println("GEMMA4_DECODE_REPLAY_DRIFT step=" + i + " token=" + continuation[i]
                            + " drift=" + driftLastBatchRow(replayOutput, decodeOutput));
                }
            }
        }
    }

    private static void printPromptAndTopK(AbstractModel model, boolean thinking) throws Exception {
        PromptContext prompt = prompt(model, thinking);
        int[] tokenizerIds = model.getTokenizer().encode(prompt.getPrompt()).inputIds();
        long[] runtimeIds = model.encodeForRuntime(prompt.getPrompt());
        int[] finalIds = model.constructPromptTokensForRuntime(prompt.getPrompt());
        System.out.println("GEMMA4_PROMPT thinking=" + thinking + " rendered=" + prompt.getPrompt().replace("\n", "\\n"));
        System.out.println("GEMMA4_TOKENIZER_IDS thinking=" + thinking + " " + java.util.Arrays.toString(tokenizerIds));
        System.out.println("GEMMA4_RUNTIME_IDS thinking=" + thinking + " " + java.util.Arrays.toString(runtimeIds));
        System.out.println("GEMMA4_FINAL_IDS thinking=" + thinking + " " + java.util.Arrays.toString(finalIds));
        System.out.println("GEMMA4_FINAL_DECODED thinking=" + thinking + " "
                + model.getTokenizer().decode(new TokenIds(finalIds), false, false, false, false).replace("\n", "\\n"));
        try (KvBufferCache.KvBuffer kv = model.newKvBuffer();
             AbstractTensor hidden = model.batchForward(finalIds, 0, kv);
             AbstractTensor logits = logits(model, hidden)) {
            System.out.println("GEMMA4_FIRST_TOP20 thinking=" + thinking + " " + topK(model, logits, 20));
        }
    }

    private static PromptContext prompt(AbstractModel model, boolean thinking) {
        return model.promptSupport().orElseThrow().builder()
                .addTemplateArgs(Map.of("enable_thinking", thinking))
                .addUserMessage("You have to choose between buying gold or buying silver. What do you buy?")
                .build();
    }

    private static void generateThree(AbstractModel model, String label, PromptContext prompt) {
        Response response = model.generate(UUID.randomUUID(), prompt,
                new GeneratorParameters().withTemperature(0.0f).withMaxTokens(3), new DoNothingGenerateEvent());
        System.out.println("GEMMA4_THINK3_" + label + "=" + response.responseTextWithSpecialTokens.replace("\n", "\\n"));
    }

    private static void printTopKForRawPrompt(AbstractModel model, String label, String prompt) throws Exception {
        int[] finalIds = model.constructPromptTokensForRuntime(prompt);
        System.out.println("GEMMA4_CHANNEL_LABEL_" + label + "_IDS=" + java.util.Arrays.toString(finalIds));
        try (KvBufferCache.KvBuffer kv = model.newKvBuffer();
             AbstractTensor hidden = model.batchForward(finalIds, 0, kv);
             AbstractTensor logits = logits(model, hidden)) {
            System.out.println("GEMMA4_CHANNEL_LABEL_" + label + "_TOP20=" + topK(model, logits, 20));
        }
    }

    private static JsonNode safetensorsHeader(File file) throws IOException {
        try (RandomAccessFile raf = new RandomAccessFile(file, "r")) {
            byte[] lengthBytes = new byte[Long.BYTES];
            raf.readFully(lengthBytes);
            long headerLength = ByteBuffer.wrap(lengthBytes).order(ByteOrder.LITTLE_ENDIAN).getLong();
            byte[] header = new byte[Math.toIntExact(headerLength)];
            raf.readFully(header);
            return OM.readTree(header);
        }
    }

    private static AbstractTensor tokenByTokenPrefill(AbstractModel model, int[] tokens, KvBufferCache.KvBuffer kvBuffer) {
        AbstractTensor output = null;
        for (int i = 0; i < tokens.length; i++) {
            if (output != null) {
                output.close();
            }
            output = model.forward(tokens[i], i, kvBuffer);
        }
        return output;
    }

    private static AbstractTensor coldReplay(AbstractModel model, int[] promptTokens, int[] continuation, int continuationLength) {
        int[] tokens = java.util.Arrays.copyOf(promptTokens, promptTokens.length + continuationLength);
        System.arraycopy(continuation, 0, tokens, promptTokens.length, continuationLength);
        return model.batchForward(tokens, 0);
    }

    private static int[] toInts(long[] values) {
        return java.util.Arrays.stream(values).mapToInt(value -> (int) value).toArray();
    }

    private static AbstractTensor logits(AbstractModel model, AbstractTensor last) throws Exception {
        SampleOutput sampleOutput = sampleOutput(model);
        ConfigurableTensorProvider provider = configurableTensorProvider(model);
        AbstractTensor logits = model.makeDenseTensor(model.getConfig().vocabularySize);
        try (AbstractTensor embedding = sampleOutput.getOutputLayerNorm().forward(last.slice(last.shape().first() - 1))) {
            provider.get().batchDotProduct(logits, embedding, sampleOutput.getOutputLogitsWeights(), 0, 0,
                    model.getConfig().embeddingLength);
        }
        return logits;
    }

    private static List<String> topK(AbstractModel model, AbstractTensor logits, int k) {
        return IntStream.range(0, logits.shape().last())
                .boxed()
                .sorted(Comparator.comparingDouble((Integer i) -> logits.get(0, i)).reversed())
                .limit(k)
                .map(i -> String.format(Locale.ROOT, "%d:%.4f:%s", i, logits.get(0, i),
                        model.getTokenizer().decode(new TokenIds(new int[]{i}), false, false, false, false)
                                .replace("\n", "\\n")))
                .toList();
    }

    private static SampleOutput sampleOutput(AbstractModel model) throws Exception {
        Field field = AbstractModel.class.getDeclaredField("sampleOutput");
        field.setAccessible(true);
        return (SampleOutput) field.get(model);
    }

    private static ConfigurableTensorProvider configurableTensorProvider(AbstractModel model) throws Exception {
        Field field = AbstractModel.class.getDeclaredField("configurableTensorProvider");
        field.setAccessible(true);
        return (ConfigurableTensorProvider) field.get(model);
    }

    private static Drift driftLastBatchRow(AbstractTensor batchOutput, AbstractTensor tokenOutput) {
        double total = 0.0;
        float max = 0.0f;
        int count = 0;
        int lastBatchRow = batchOutput.shape().first() - 1;
        for (int col = 0; col < batchOutput.shape().last(); col++) {
            float diff = Math.abs(batchOutput.get(lastBatchRow, col) - tokenOutput.get(0, col));
            if (diff > max) {
                max = diff;
            }
            total += diff;
            count++;
        }
        return new Drift(max, count == 0 ? 0.0 : total / count, count);
    }

    private record Drift(float maxAbs, double meanAbs, int values) {
    }
}
