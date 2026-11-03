package io.teknek.reckoning.model;

import io.teknek.reckoning.safetensors.fetch.ModelFetcher;
import io.teknek.reckoning.tensor.KvBufferCacheSettings;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AutoModelForCausaLmPrefixCachePolicyTest {

    @Test
    void qwen06bRejectsTurboQuantPrefixCacheInBuilder() {
        KvBufferCacheSettings settings = new KvBufferCacheSettings()
                .withPrefixCompression(KvBufferCacheSettings.PrefixCompression.MSE_TURBOQUANT);

        IllegalArgumentException thrown = assertThrows(IllegalArgumentException.class,
                () -> AutoModelForCausaLm.newBuilder(new ModelFetcher("edwardcapriolo", "Qwen3-0.6B-JQ4"))
                        .withKvBufferCacheSettings(settings));

        assertTrue(thrown.getMessage().contains("TurboQuant degrades with super small models"));
    }

    @Test
    void qwen06bAllowsExactPrefixCacheInBuilder() {
        KvBufferCacheSettings settings = new KvBufferCacheSettings()
                .withPrefixCompression(KvBufferCacheSettings.PrefixCompression.NONE);

        assertDoesNotThrow(() -> AutoModelForCausaLm.newBuilder(new ModelFetcher("edwardcapriolo", "Qwen3-0.6B-JQ4"))
                .withKvBufferCacheSettings(settings));
    }
}
