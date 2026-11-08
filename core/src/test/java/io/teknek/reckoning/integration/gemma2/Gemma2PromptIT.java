package io.teknek.reckoning.integration.gemma2;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.reckoning.JsonUtils;
import io.teknek.reckoning.generator.GeneratorParameters;
import io.teknek.reckoning.generator.Response;
import io.teknek.reckoning.model.AbstractModel;
import io.teknek.reckoning.model.AutoModelForCausaLm;
import io.teknek.reckoning.model.DoNothingGenerateEvent;
import io.teknek.reckoning.safetensors.fetch.ModelFetcher;
import io.teknek.reckoning.safetensors.prompt.PromptContext;
import io.teknek.reckoning.safetensors.prompt.PromptSupport;
import io.teknek.reckoning.DType;
import io.teknek.reckoning.tensor.KvBufferCacheSettings;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class Gemma2PromptIT {

    @Test
    public void chat() {
        AbstractModel model = Gemma2Suite.getOrCreate();
        String prompt = """
                What does this python code do?
                ---------------------------
                def allocate_token_bitmask(vocab_size: int) -> np.ndarray:
                    return np.full(
                        (1, (vocab_size + 31) // 32),
                        -1,
                        dtype=np.int32,
                    )
                """;
        PromptSupport.Builder builder = model.promptSupport().orElseThrow().builder()
                .addUserMessage(prompt);
        Response response = model.generate(UUID.randomUUID(), builder.build(),
                new GeneratorParameters().withMaxTokens(100).withTemperature(0.0f),
                new DoNothingGenerateEvent());
        System.out.println(response.responseText);
    }

    @Test
    public void gemmaTest() {
        AbstractModel model = Gemma2Suite.getOrCreate();
        MetricRegistry metrics = Gemma2Suite.getBuilder().getMr();
        String prompt = "What is the capital of New York, USA?";
        PromptSupport.Builder builder = model.promptSupport().orElseThrow().builder()
                .addUserMessage(prompt);
        assertEquals("<bos><start_of_turn>user\n" +
                "What is the capital of New York, USA?<end_of_turn>\n" +
                "<start_of_turn>model\n", builder.build().getPrompt());

        Response response = model.generate(UUID.randomUUID(), builder.build(),
                new GeneratorParameters().withTemperature(0.0f),
                new DoNothingGenerateEvent());
        System.out.println(response.responseText);
        assertTrue(response.responseText.contains("Albany"));
        System.out.println(Arrays.toString(metrics.histogram("sample.fullsample").getSnapshot().getValues()));
        System.out.println(metrics.histogram("sample.fullsample").getSnapshot().getMean());
        System.out.println(metrics.histogram("sample.fullsample").getSnapshot().get99thPercentile());
        System.out.println(Arrays.toString(metrics.histogram("sample.forward1").getSnapshot().getValues()));
        System.out.println(metrics.histogram("sample.forward1").getSnapshot().getMean());
        System.out.println(metrics.histogram("sample.forward1").getSnapshot().get99thPercentile());
        System.out.println(Arrays.toString(metrics.histogram("sample.dotproduct2").getSnapshot().getValues()));
        System.out.println(metrics.histogram("sample.dotproduct2").getSnapshot().getMean());
        System.out.println(metrics.histogram("sample.dotproduct2").getSnapshot().get99thPercentile());
    }

    @Test
    public void gemmaAlbanyFirstTokens() {
        AbstractModel model = Gemma2Suite.getOrCreate();
        PromptContext prompt = model.promptSupport().orElseThrow().builder()
                .addUserMessage("What is the capital of New York, USA?").build();
        Response response = model.generate(UUID.randomUUID(), prompt, new GeneratorParameters()
                        .withMaxTokens(3).withTemperature(0.0f).withSeed(123),
                new DoNothingGenerateEvent());
        System.out.printf("GEMMA_SUITE_RESULT tokens=%s text=%s special=%s%n",
                response.generatedTokens, response.responseText, response.responseTextWithSpecialTokens);
         assertEquals(List.of(1596, 603, 476), response.generatedTokens, "Gemma2 Albany first tokens");
    }

    @Test
    public void packedPrefillMatchesPagedAttentionForFirstGemmaTokens() {
        ModelFetcher fetch = new ModelFetcher("tjake", "gemma-2-2b-it-JQ4");
        try (AbstractModel pagedModel = AutoModelForCausaLm.newBuilder(fetch)
                     .withOutputHeadQuantization(DType.Q4)
                     .withKvBufferCacheSettings(new KvBufferCacheSettings().withBlockSize(8))
                     .withPackedPrefill(false).buildLocalTransformerModel();
             AbstractModel packedModel = AutoModelForCausaLm.newBuilder(fetch)
                     .withOutputHeadQuantization(DType.Q4)
                     .withKvBufferCacheSettings(new KvBufferCacheSettings().withBlockSize(8))
                     .withPackedPrefill(true).buildLocalTransformerModel()) {
            PromptContext pagedPrompt = pagedModel.promptSupport().orElseThrow().builder()
                    .addUserMessage("What is the capital of New York, USA?").build();
            PromptContext packedPrompt = packedModel.promptSupport().orElseThrow().builder()
                    .addUserMessage("What is the capital of New York, USA?").build();
            GeneratorParameters parameters = new GeneratorParameters().withMaxTokens(2)
                    .withTemperature(0.0f).withSeed(123);
            Response paged = pagedModel.generate(UUID.randomUUID(), pagedPrompt, parameters,
                    new DoNothingGenerateEvent());
            Response packed = packedModel.generate(UUID.randomUUID(), packedPrompt, parameters,
                    new DoNothingGenerateEvent());
            assertEquals(paged.generatedTokens, packed.generatedTokens,
                    "packed prefill should choose the same first Gemma2 tokens as paged attention");
            assertEquals(paged.responseText, packed.responseText,
                    "packed prefill should produce the same first Gemma2 text as paged attention");
        }
    }

    @Test
    public void gemmaGuidedTest() {
        AbstractModel model = Gemma2Suite.getOrCreate();
        PromptSupport.Builder builder = model.promptSupport().orElseThrow().builder()
                .addUserMessage("Who is the better NFL football team?");
        assertEquals("<bos><start_of_turn>user\n" +
                "Who is the better NFL football team?<end_of_turn>\n" +
                "<start_of_turn>model\n", builder.build().getPrompt());
        Response response = model.generate(UUID.randomUUID(), builder.build(), new GeneratorParameters()
                        .withTemperature(0.0f).withGuidedChoice(List.of("Giants", "Jets")),
                new DoNothingGenerateEvent());
        System.out.println(response.responseText);
        assertTrue(response.responseText.contains("Giants"));
    }

    @Test
    public void gemmaGuidedRegexTest() {
        AbstractModel model = Gemma2Suite.getOrCreate();
        PromptSupport.Builder builder = model.promptSupport().orElseThrow().builder()
                .addUserMessage("Who is the better NFL football team?");
        Response response = model.generate(UUID.randomUUID(), builder.build(), new GeneratorParameters()
                        .withTemperature(0.0f).withGuidedRegex("(Giants|Jets)"),
                new DoNothingGenerateEvent());
        assertTrue(response.responseText.contains("Giants") || response.responseText.contains("Jets"));
    }

    @Test
    public void gemmaGuidedJsonTest() throws Exception {
        AbstractModel model = Gemma2Suite.getOrCreate();
        String schema = """
                {
                  "type": "object",
                  "properties": { "answer": { "const": "Giants" } },
                  "required": ["answer"],
                  "additionalProperties": false
                }
                """;
        PromptSupport.Builder builder = model.promptSupport().orElseThrow().builder()
                .addUserMessage("Return the answer as JSON.");
        Response response = model.generate(UUID.randomUUID(), builder.build(), new GeneratorParameters()
                        .withTemperature(0.0f).withMaxTokens(30).withGuidedJson(schema),
                new DoNothingGenerateEvent());
        assertEquals("Giants", JsonUtils.om.readTree(response.responseText).get("answer").asText());
        assertEquals("{\"answer\":\"Giants\"}", response.responseText);
    }

    @Test
    public void gemmaGuidedTestNeg() {
        AbstractModel model = Gemma2Suite.getOrCreate();
        PromptSupport.Builder builder = model.promptSupport().orElseThrow().builder()
                .addUserMessage("Which NFL franchise does not play in New York?");
        Response response = model.generate(UUID.randomUUID(), builder.build(), new GeneratorParameters()
                        .withTemperature(0.0f)
                        .withGuidedChoice(List.of("Giants", "Jets", "Seahawks")),
                new DoNothingGenerateEvent());
        assertTrue(response.responseText.contains("Jets"));
    }

    @Test
    public void xtc() {
        AbstractModel model = Gemma2Suite.getOrCreate();
        PromptSupport.Builder builder = model.promptSupport().orElseThrow().builder()
                .addUserMessage("Pick a random number between 1 and 9.");
        Response response = model.generate(UUID.randomUUID(), builder.build(), new GeneratorParameters()
                        .withTemperature(0.0f).withMaxTokens(300).withLogProbs(true)
                        .withTopLogProbs(10).withXtcThreshold(0.1f).withXtcProbability(0.5f),
                new DoNothingGenerateEvent());
        System.out.println(response.responseText);
    }

    @Test
    public void topK() {
        generateSampling(new GeneratorParameters().withTemperature(0.3f).withMaxTokens(20)
                .withTopK(0.9999f), "Reply with only the pick.");
    }

    @Test
    public void temperatureAndXtc() {
        generateSampling(new GeneratorParameters().withTemperature(0.3f).withMaxTokens(20)
                .withLogProbs(true).withTopLogProbs(10).withXtcThreshold(0.1f)
                .withXtcProbability(0.5f), "");
    }

    @Test
    public void temperatureAndTopp() {
        generateSampling(new GeneratorParameters().withTemperature(0.9f).withMaxTokens(20)
                .withLogProbs(true).withTopLogProbs(10).withTopP(0.2f),
                "Reply with only the pick.");
    }

    private void generateSampling(GeneratorParameters parameters, String suffix) {
        AbstractModel model = Gemma2Suite.getOrCreate();
        PromptSupport.Builder builder = model.promptSupport().orElseThrow().builder()
                .addUserMessage("Pick a random number between 1 and 9. " + suffix);
        Response response = model.generate(UUID.randomUUID(), builder.build(), parameters,
                new DoNothingGenerateEvent());
        System.out.println(response.responseText);
    }

    // The legacy summarizeGemmaTest and logProbs methods were disabled exploratory code,
    // not enabled suite coverage; their large ad hoc output fixtures are intentionally omitted.
}
