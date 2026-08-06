package io.teknek.deliverance.integration;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.model.DoNothingGenerateEvent;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.stream.Stream;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
//this IT shows that you cant use TurboQuantPrefix on a very small model as it collapses
class Qwen06bTurboQuantPrefixCacheIT {

    @ParameterizedTest(name = "{0}/{1}")
    @MethodSource("turboQuantPrefixCases")
    void qwenTurboQuantPrefixCacheHitProducesExpectedLossyOutput(String owner, String modelName,
            String expectedCold, String expectedHot) {
        ModelFetcher fetch = new ModelFetcher(owner, modelName).withDownload(false);
        Assumptions.assumeTrue(fetch.pathForModel().toFile().isDirectory(),
                modelName + " cache is not present: " + fetch.pathForModel());

        KvBufferCacheSettings settings = new KvBufferCacheSettings(true)
                .withMaxEntries(64)
                .withBlockSize(8)
                .withMaxPrefixTokensPerPrompt(512)
                .withMaxPrefixCheckpointsPerPrompt(8)
                .withPrefixCheckpointPolicy(KvBufferCacheSettings.PrefixCheckpointPolicy.FIXED_BLOCKS)
                .withPrefixCompression(KvBufferCacheSettings.PrefixCompression.MSE_TURBOQUANT)
                .withPrefixTurboQuantBits(4);

        try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetch)
                .withDownload(false)
                .withWorkingQuantType(DType.I8)
                .withKvBufferCacheSettings(settings)
                .buildLocalTransformerModel()) {
            PromptContext prompt = model.promptSupport().orElseThrow().builder()
                    .addTemplateArgs(java.util.Map.of("enable_thinking", false))
                    .addSystemMessage("You are a concise assistant. Answer plainly.")
                    .addUserMessage("Name ten ordinary places someone might visit in a small town, then stop.")
                    .build();
            GeneratorParameters params = new GeneratorParameters()
                    .withTemperature(0.0f)
                    .withSeed(42)
                    .withMaxTokens(32)
                    .withCacheSalt("qwen06b-turboquant-prefix-" + UUID.randomUUID());

            Response cold = model.generate(UUID.randomUUID(), prompt, params, new DoNothingGenerateEvent());
            long hitsBefore = model.getMetricRegistry().meter("kvbuffercache.hits").getCount();
            AtomicInteger copiedPrefixLength = new AtomicInteger(-1);
            AtomicInteger suffixLength = new AtomicInteger(-1);
            model.setGenerationDebugHook(event -> {
                if (event.type() == AbstractModel.GenerationDebugEventType.AFTER_PREFIX_COPY) {
                    copiedPrefixLength.set(event.prefixLength());
                    suffixLength.set(event.tokensToProcessLength());
                }
            });
            Response hot = model.generate(UUID.randomUUID(), prompt, params, new DoNothingGenerateEvent());
            model.clearGenerationDebugHook();

            System.out.println("QWEN_TURBO_PREFIX_MODEL=" + owner + "/" + modelName);
            System.out.println("QWEN_TURBO_PREFIX_COLD=" + cold.responseText.replace("\n", "\\n"));
            System.out.println("QWEN_TURBO_PREFIX_HOT=" + hot.responseText.replace("\n", "\\n"));
            System.out.println("QWEN_TURBO_PREFIX_LENGTH=" + copiedPrefixLength.get()
                    + " SUFFIX=" + suffixLength.get());

            assertTrue(model.getMetricRegistry().meter("kvbuffercache.hits").getCount() > hitsBefore,
                    "second request should hit prefix cache");
            assertTrue(copiedPrefixLength.get() > 0, "prefix hit should copy some prefix rows");
            assertEquals(0, copiedPrefixLength.get() % settings.getBlockSize(),
                    "prefix hit should be block aligned");
            assertFalse(cold.responseText.isBlank(), "cold generation should produce text");
            assertFalse(hot.responseText.isBlank(), "hot generation should produce text");
            if (expectedCold != null) {
                assertEquals(expectedCold, cold.responseText,
                        "cold deterministic output changed; update this characterization only after reviewing model changes");
            }
            if (expectedHot != null) {
                assertEquals(expectedHot, hot.responseText,
                        "TurboQuant prefix cache lossy output changed; update this characterization only after reviewing cache changes");
            }
        }
    }

    private static Stream<Arguments> turboQuantPrefixCases() {
        return Stream.of(
                Arguments.of(
                        "edwardcapriolo",
                        "Qwen3-0.6B-JQ4",
                        "1. **The Post Office** – A small town might have a post office, and visiting there is a great way to experience the town's charm.",
                        "1. you can visit it,, ,,, and you visit it, it,, and it is a small town. you visit it,,"
                ),
                Arguments.of(
                        "edwardcapriolo",
                        "Qwen3-4B-JQ4",
                        "1. Café  \n2. Grocery store  \n3. Library  \n4. Post office  \n5. Dentist's office  \n6. Doctor's office  \n7",
                        "1. Café  \n2. Library  \n3. Grocery store  \n4. Post office  \n5. Dentist  \n6. Doctor  \n7. Hair salon"
                )
        );
    }
}
