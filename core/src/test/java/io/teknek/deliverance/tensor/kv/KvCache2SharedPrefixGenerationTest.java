package io.teknek.deliverance.tensor.kv;

import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.model.DoNothingGenerateEvent;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class KvCache2SharedPrefixGenerationTest {

    @Test
    public void qwen306BRepeatedPromptReusesSharedKvCache2PrefixBlocks() {
        int blockSize = 8;
        KvBufferCacheSettings settings = new KvBufferCacheSettings(true)
                .withBlockSize(blockSize)
                .withMaxPrefixTokensPerPrompt(256)
                .withPrefixCacheMode(KvBufferCacheSettings.PrefixCacheMode.SHARED_BLOCKS);
        ModelFetcher fetch = new ModelFetcher("edwardcapriolo", "Qwen3-0.6B-JQ4");
        try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetch)
                .withKvBufferCacheSettings(settings)
                .buildLocalTransformerModel()) {
            PromptContext prompt = model.promptSupport().orElseThrow().builder()
                    .addTemplateArgs(Map.of("enable_thinking", false))
                    .addSystemMessage("You are a concise assistant. Answer plainly.")
                    .addUserMessage("Name five ordinary objects someone might find on a kitchen table.")
                    .build();
            GeneratorParameters parameters = new GeneratorParameters()
                    .withTemperature(0.0f)
                    .withSeed(42)
                    .withMaxTokens(8)
                    .withCacheSalt("qwen3-small-shared-kv2-prefix");

            Response cold = model.generate(UUID.randomUUID(), prompt, parameters, new DoNothingGenerateEvent());
            long admittedBeforeHot = model.getMetricRegistry()
                    .counter("kvcache.v2.prefix.blocks.admitted").getCount();
            long attachedBeforeHot = model.getMetricRegistry()
                    .counter("kvcache.v2.prefix.blocks.attached").getCount();
            long reusedBeforeHot = model.getMetricRegistry()
                    .counter("kvcache.v2.prefix.tokens.reused").getCount();
            AtomicInteger copiedPrefixLength = new AtomicInteger(0);
            model.setGenerationDebugHook(event -> {
                if (event.type() == AbstractModel.GenerationDebugEventType.AFTER_PREFIX_COPY) {
                    copiedPrefixLength.set(event.prefixLength());
                }
            });

            Response hot;
            try {
                hot = model.generate(UUID.randomUUID(), prompt, parameters, new DoNothingGenerateEvent());
            } finally {
                model.clearGenerationDebugHook();
            }

            System.out.println("QWEN3_06B_KV2_PREFIX_COLD=" + cold.responseTextWithSpecialTokens.replace("\n", "\\n"));
            System.out.println("QWEN3_06B_KV2_PREFIX_HOT=" + hot.responseTextWithSpecialTokens.replace("\n", "\\n"));
            System.out.println("QWEN3_06B_KV2_PREFIX_LENGTH=" + copiedPrefixLength.get());
            assertFalse(cold.responseTextWithSpecialTokens.isBlank());
            assertFalse(hot.responseTextWithSpecialTokens.isBlank());
            assertTrue(admittedBeforeHot > 0, "cold generation should admit shared KVCache2 prefix blocks");
            assertTrue(model.getMetricRegistry().counter("kvcache.v2.prefix.blocks.attached").getCount()
                    > attachedBeforeHot, "hot generation should attach shared KVCache2 prefix blocks");
            assertTrue(model.getMetricRegistry().counter("kvcache.v2.prefix.tokens.reused").getCount()
                    > reusedBeforeHot, "hot generation should reuse shared KVCache2 prefix tokens");
            assertTrue(copiedPrefixLength.get() >= blockSize, "hot generation should report a copied KV2 prefix");
            assertEquals(0, copiedPrefixLength.get() % blockSize, "KV2 prefix reuse must be block aligned");
        }
    }
}
