package io.teknek.deliverance.integration;


import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.generator.SampleOutput;
import io.teknek.deliverance.grace.TokenIds;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.DoNothingGenerateEvent;
import io.teknek.deliverance.model.GenerateEvent;
import io.teknek.deliverance.model.gemma4.Gemma4ResponseParser;
import io.teknek.deliverance.safetensors.prompt.Function;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import io.teknek.deliverance.safetensors.prompt.Tool;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.KvBufferCache;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class Gemma4PromptIT {
    private static final boolean DEBUG_PROMPTS = Boolean.getBoolean("deliverance.gemma4.prompt.debug");

    @Disabled
    public void chatWithThinking() {

        AbstractModel model = Gemma4Suite.getOrCreate();
        PromptSupport.Builder builder = model.promptSupport().get().builder()
                .addTemplateArgs(Map.of("enable_thinking", false))
                /*
                .addUserMessage("""
                        You MUST perform reasoning before writing the reply.
                        Your internal thought process MUST be generated BEFORE any final answer.
                        Show your reasoning in <think> tags, then provide the final, concise answer.
                        Take your time and evaluate multiple alternative explanations.
                        You have to choose between buying gold or buying silver. What do you buy?
                        """);*/
                .addUserMessage("""
                        
                            You have to choose between buying gold or buying silver. What do you buy?
                        """);
        PromptContext promptContext = builder.build();
        debugPrompt(model, promptContext);

        Response response = model.generate(
                UUID.randomUUID(),
                promptContext,
                new GeneratorParameters().withTemperature(0.0f)
                        /*.withLogProbs(true).withTopLogProbs(10)*/.withMaxTokens(90),
                new GenerateEvent() {
                    @Override
                    public void emit(int next, String nextRaw, String nextCleaned, float timing) {
                        System.out.println(next + " " + nextCleaned);
                    }
                }
        );
        Gemma4ResponseParser.Parsed parsed = Gemma4ResponseParser.parse(
                response.responseTextWithSpecialTokens,
                response.responseText
        );
        assertTrue(parsed.reasoning() == null,
                "Thinking was disabled, so reasoning should not be parsed: " + response.responseTextWithSpecialTokens);
    }

    @Test
    public void chatWithToolTemplate() {
        AbstractModel model = Gemma4Suite.getOrCreate();
        Tool tool = Tool.from(
                Function.builder()
                        .name("get_weather")
                        .description("Gets the current weather.")
                        .addParameter("location", "string", "City and state.", true)
                        .build()
        );

        PromptContext promptContext = model.promptSupport().orElseThrow().builder()
                .addToolItem(tool)
                .addUserMessage("What is the weather in Albany, NY?")
                .build();

        assertEquals("""
                <|turn>system
                <|tool>declaration:get_weather{description:<|"|>Gets the current weather.<|"|>,parameters:{properties:{location:{description:<|"|>City and state.<|"|>,type:<|"|>STRING<|"|>}},required:[<|"|>location<|"|>],type:<|"|>OBJECT<|"|>}}<tool|><turn|>
                <|turn>user
                What is the weather in Albany, NY?<turn|>
                <|turn>model
                """, promptContext.getPrompt());
    }

    @Disabled
    public void batchPrefillMatchesTokenByTokenPrefill() {
        AbstractModel model = Gemma4Suite.getOrCreate();
        PromptContext promptContext = model.promptSupport().orElseThrow().builder()
                .addTemplateArgs(Map.of("enable_thinking", false))
                .addUserMessage("You have to choose between buying gold or buying silver. What do you buy?")
                .build();
        int[] promptTokens = model.constructPromptTokensForRuntime(promptContext.getPrompt());

        try (KvBufferCache.KvBuffer batchKv = model.newKvBuffer();
             KvBufferCache.KvBuffer stepKv = model.newKvBuffer();
             AbstractTensor batchOutput = model.batchForward(promptTokens, 0, batchKv);
             AbstractTensor stepOutput = tokenByTokenPrefill(model, promptTokens, stepKv)) {
            Drift drift = driftLastBatchRow(batchOutput, stepOutput);
            System.out.printf(java.util.Locale.ROOT,
                    "gemma4 batch-vs-token prefill drift max_abs=%.6f mean_abs=%.6f values=%d%n",
                    drift.maxAbs(), drift.meanAbs(), drift.values());
            assertTrue(drift.maxAbs() < 1.0f,
                    "Gemma4 batch prefill diverged from token-by-token prefill: " + drift);
        }
    }

    @Disabled
    public void decodeHiddenMatchesColdReplayForFixedContinuation() {
        AbstractModel model = Gemma4Suite.getOrCreate();
        int[] promptTokens = promptTokens(model);
        int[] continuation = ints(model.encodeForRuntime(" This is a classic overthought scenario question."));

        try (KvBufferCache.KvBuffer decodeKv = model.newKvBuffer();
             AbstractTensor promptOutput = model.batchForward(promptTokens, 0, decodeKv)) {
            promptOutput.close();
            for (int i = 0; i < continuation.length; i++) {
                int position = promptTokens.length + i;
                try (AbstractTensor decodeOutput = model.forward(continuation[i], position, decodeKv);
                     AbstractTensor replayOutput = coldReplay(model, promptTokens, continuation, i + 1)) {
                    Drift drift = driftLastBatchRow(replayOutput, decodeOutput);
                    System.out.printf(java.util.Locale.ROOT,
                            "gemma4 decode-vs-cold hidden step=%d token=%d max_abs=%.6f mean_abs=%.6f%n",
                            i, continuation[i], drift.maxAbs(), drift.meanAbs());
                    assertTrue(drift.maxAbs() < 1.0f,
                            "Gemma4 decode hidden diverged from cold replay at step " + i + ": " + drift);
                }
            }
        }
    }

    @Disabled
    public void decodeLogitsMatchColdReplayForFixedContinuation() throws Exception {
        AbstractModel model = Gemma4Suite.getOrCreate();
        int[] promptTokens = promptTokens(model);
        int[] continuation = ints(model.encodeForRuntime(" This is a classic overthought scenario question."));

        try (KvBufferCache.KvBuffer decodeKv = model.newKvBuffer();
             AbstractTensor promptOutput = model.batchForward(promptTokens, 0, decodeKv)) {
            promptOutput.close();
            for (int i = 0; i < continuation.length; i++) {
                int position = promptTokens.length + i;
                try (AbstractTensor decodeOutput = model.forward(continuation[i], position, decodeKv);
                     AbstractTensor replayOutput = coldReplay(model, promptTokens, continuation, i + 1);
                     AbstractTensor decodeLogits = logits(model, decodeOutput);
                     AbstractTensor replayLogits = logits(model, replayOutput)) {
                    int decodeArgmax = argmax(decodeLogits);
                    int replayArgmax = argmax(replayLogits);
                    int overlap = topKOverlap(decodeLogits, replayLogits, 50);
                    System.out.printf(java.util.Locale.ROOT,
                            "gemma4 decode-vs-cold logits step=%d token=%d decode_argmax=%d replay_argmax=%d top50_overlap=%d%n",
                            i, continuation[i], decodeArgmax, replayArgmax, overlap);
                    assertEquals(replayArgmax, decodeArgmax,
                            "Gemma4 decode argmax diverged from cold replay at step " + i);
                    assertTrue(overlap >= 45,
                            "Gemma4 decode top50 diverged from cold replay at step " + i + ": overlap=" + overlap);
                }
            }
        }
    }

    @Test
    public void traceKnownBadContinuation() throws Exception {
        Assumptions.assumeTrue(Boolean.getBoolean("deliverance.gemma4.badtrace"),
                "Set -Ddeliverance.gemma4.badtrace=true to trace the known bad continuation");
        AbstractModel model = Gemma4Suite.getOrCreate();
        int[] promptTokens = promptTokens(model);
        int[] knownBadContinuation = new int[] {
                2094, 563, 496, 9760, 236772, 23461, 62557, 16613, 236764, 6111,
                872, 684, 506, 9639, 66752, 64121, 2921, 532, 26806, 203902,
                495, 236786, 47496, 25055, 65006, 1072, 124336, 236754, 8218,
                206425, 532, 3415, 121762, 109236, 149180, 147839, 13897,
                34081, 532, 26806, 501, 15595, 22100, 13994, 743, 15823,
                7086, 79005, 115743, 11960, 502, 527, 61315, 236747, 236772,
                150693, 19995, 21294, 28660, 16012, 236747, 23141, 43174,
                34054, 79492, 14930, 508, 1499
        };

        try (KvBufferCache.KvBuffer kv = model.newKvBuffer();
             AbstractTensor promptOutput = model.batchForward(promptTokens, 0, kv)) {
            promptOutput.close();
            for (int i = 0; i < knownBadContinuation.length; i++) {
                int expectedToken = knownBadContinuation[i];
                int position = promptTokens.length + i;
                try (AbstractTensor previous = i == 0
                        ? coldReplay(model, promptTokens, new int[0], 0)
                        : model.forward(knownBadContinuation[i - 1], position - 1, kv);
                     AbstractTensor stepLogits = logits(model, previous)) {
                    int argmax = argmax(stepLogits);
                    int expectedRank = rank(stepLogits, expectedToken);
                    List<Integer> top10 = topK(stepLogits, 10);
                    System.out.printf(java.util.Locale.ROOT,
                            "gemma4 known_bad_trace step=%d expected=%d argmax=%d expected_rank=%d expected_logit=%.6f argmax_logit=%.6f top10=%s%n",
                            i, expectedToken, argmax, expectedRank, stepLogits.get(0, expectedToken),
                            stepLogits.get(0, argmax), top10);
                }
            }
        }
    }

    @Disabled
    public void chatWithReasoning() {
        AbstractModel model = Gemma4Suite.getOrCreate();
        PromptSupport.Builder builder = model.promptSupport().get().builder()
                .addTemplateArgs(Map.of("enable_thinking", true))
                //.addUserMessage("Bob is a carpenter. Sara is a teacher. Who should you call to fix your roof? Pick one of the two. Only answer one name. ");
                .addUserMessage("""
                        You MUST perform reasoning before writing the reply.\s
                        Your internal thought process MUST be generated BEFORE any final answer.\s
                        Show your reasoning in <think> tags, then provide the final, concise answer.\s
                        Take your time and evaluate multiple alternative explanations.
                        You have to chose between buying gold or buying silver. What do you buy?
                        """);
        PromptContext promptContext = builder.build();
        Assertions.assertTrue(promptContext.toString().contains("<|think|>"));
        Response response = model.generate(
                UUID.randomUUID(),
                promptContext,
                new GeneratorParameters().withTemperature(0.0f).withMaxTokens(90),
                new GenerateEvent() {
                    @Override
                    public void emit(int next, String nextRaw, String nextCleaned, float timing) {
                        System.out.println(next + " " + nextCleaned);
                    }
                }
        );
        System.out.println(response);
    }


    private static void debugPrompt(AbstractModel model, PromptContext promptContext) {
        if (!DEBUG_PROMPTS) {
            return;
        }
        int[] promptTokens = model.getTokenizer().encode(promptContext.getPrompt()).inputIds();
        long[] runtimePromptTokens = model.encodeForRuntime(promptContext.getPrompt());
        int[] finalPromptTokens = model.constructPromptTokensForRuntime(promptContext.getPrompt());
        System.out.println("PROMPT_RENDERED_START");
        System.out.println(promptContext.getPrompt());
        System.out.println("PROMPT_RENDERED_END");
        System.out.println("PROMPT_TOKEN_IDS=" + Arrays.toString(promptTokens));
        System.out.println("PROMPT_DECODED_START");
        System.out.println(model.getTokenizer().decode(new TokenIds(promptTokens), false, false, false, false));
        System.out.println("PROMPT_DECODED_END");
        System.out.println("RUNTIME_PROMPT_TOKEN_IDS=" + Arrays.toString(runtimePromptTokens));
        System.out.println("RUNTIME_PROMPT_DECODED_START");
        System.out.println(model.getTokenizer().decode(new TokenIds(Arrays.stream(runtimePromptTokens).mapToInt(v -> (int) v).toArray()), false, false, false, false));
        System.out.println("RUNTIME_PROMPT_DECODED_END");
        System.out.println("FINAL_PROMPT_TOKEN_IDS=" + Arrays.toString(finalPromptTokens));
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

    private static int[] promptTokens(AbstractModel model) {
        PromptContext promptContext = model.promptSupport().orElseThrow().builder()
                .addTemplateArgs(Map.of("enable_thinking", false))
                .addUserMessage("You have to choose between buying gold or buying silver. What do you buy?")
                .build();
        return model.constructPromptTokensForRuntime(promptContext.getPrompt());
    }

    private static int[] ints(long[] values) {
        return Arrays.stream(values).mapToInt(value -> (int) value).toArray();
    }

    private static AbstractTensor coldReplay(AbstractModel model, int[] promptTokens, int[] continuation, int continuationLength) {
        int[] tokens = Arrays.copyOf(promptTokens, promptTokens.length + continuationLength);
        System.arraycopy(continuation, 0, tokens, promptTokens.length, continuationLength);
        return model.batchForward(tokens, 0);
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

    private static int argmax(AbstractTensor logits) {
        int best = 0;
        float bestValue = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < logits.shape().last(); i++) {
            float value = logits.get(0, i);
            if (value > bestValue) {
                best = i;
                bestValue = value;
            }
        }
        return best;
    }

    private static int topKOverlap(AbstractTensor first, AbstractTensor second, int k) {
        List<Integer> firstTop = topK(first, k);
        List<Integer> secondTop = topK(second, k);
        return (int) firstTop.stream().filter(secondTop::contains).count();
    }

    private static List<Integer> topK(AbstractTensor logits, int k) {
        return IntStream.range(0, logits.shape().last())
                .boxed()
                .sorted(Comparator.comparingDouble((Integer i) -> logits.get(0, i)).reversed())
                .limit(k)
                .toList();
    }

    private static int rank(AbstractTensor logits, int token) {
        float value = logits.get(0, token);
        int rank = 1;
        for (int i = 0; i < logits.shape().last(); i++) {
            if (logits.get(0, i) > value) {
                rank++;
            }
        }
        return rank;
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
        assertEquals(1, tokenOutput.shape().first());
        assertEquals(batchOutput.shape().last(), tokenOutput.shape().last());
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
