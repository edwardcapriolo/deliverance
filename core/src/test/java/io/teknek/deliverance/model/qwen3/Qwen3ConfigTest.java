package io.teknek.deliverance.model.qwen3;

import io.teknek.deliverance.math.ActivationFunction;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class Qwen3ConfigTest {

    @Test
    public void defaultsKeyValueHeadsToAttentionHeads() {
        Qwen3Config config = config(null, false, null, null);

        assertEquals(4, config.numberOfKeyValueHeads);
        assertEquals(4, config.numberOfHeads);
    }

    @Test
    public void defaultsLayerTypesToFullAttentionWhenSlidingWindowDisabled() {
        Qwen3Config config = config(2, false, 16, null);

        assertEquals(List.of("full_attention", "full_attention", "full_attention", "full_attention"), config.layerTypes);
        assertEquals(null, config.slidingWindow);
    }

    @Test
    public void usesSlidingAttentionAfterMaxWindowLayersWhenEnabled() {
        Qwen3Config config = config(2, true, 16, null);

        assertEquals(List.of("full_attention", "full_attention", "sliding_attention", "sliding_attention"), config.layerTypes);
        assertEquals(16, config.slidingWindow);
    }

    @Test
    public void parsesRopeThetaFromRopeParameters() {
        Qwen3Config config = config(2, false, null, Map.of("rope_type", "default", "rope_theta", 12345.0));

        assertEquals(12345.0, ((Number) config.ropeParameters.get("rope_theta")).doubleValue());
    }

    @Test
    public void parsesOfficialQwen306BDenseConfig() throws Exception {
        Path configPath = Path.of(System.getProperty("user.home"), ".deliverance", "Qwen_Qwen3-0.6B", "config.json");
        org.junit.jupiter.api.Assumptions.assumeTrue(java.nio.file.Files.isRegularFile(configPath),
                "Run HfQwen3TokenizerSmokeTest or fetch tokenizer config first");

        Qwen3Config config = io.teknek.deliverance.JsonUtils.om.readValue(configPath.toFile(), Qwen3Config.class);

        assertEquals(1024, config.embeddingLength);
        assertEquals(3072, config.hiddenLength);
        assertEquals(16, config.numberOfHeads);
        assertEquals(8, config.numberOfKeyValueHeads);
        assertEquals(28, config.numberOfLayers);
        assertEquals(128, config.headSize);
        assertEquals(151643, config.bosToken);
        assertEquals(List.of(151645), config.eosTokens);
        assertEquals(List.of("full_attention").stream().findFirst().orElseThrow(), config.layerTypes.getFirst());
        assertEquals(null, config.slidingWindow);
    }

    private static Qwen3Config config(Integer maxWindowLayers, boolean useSlidingWindow, Integer slidingWindow,
            Map<String, Object> ropeParameters) {
        return new Qwen3Config(
                32,
                16,
                32,
                4,
                null,
                4,
                1.0e-6f,
                64,
                null,
                2,
                ActivationFunction.Type.SILU,
                10_000.0,
                ropeParameters,
                4,
                useSlidingWindow,
                slidingWindow,
                maxWindowLayers,
                null,
                0.0f,
                List.of("Qwen3ForCausalLM")
        );
    }
}
