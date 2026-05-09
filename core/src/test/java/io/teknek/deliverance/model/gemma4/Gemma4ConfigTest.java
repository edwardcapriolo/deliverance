package io.teknek.deliverance.model.gemma4;

import org.junit.jupiter.api.Test;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class Gemma4ConfigTest {
    @Test
    public void fullAttentionUsesDefaultKvHeadsWhenAttentionKDoesNotEqualV() {
        Gemma4Config config = config(false);
        assertEquals(8, config.getLayerHeadDim("full_attention"));
        assertEquals(16, config.getLayerQueryProjectionLength("full_attention"));
        assertEquals(1, config.getLayerKeyValueProjectionHeads("full_attention"));
        assertEquals(8, config.getLayerKeyValueProjectionLength("full_attention"));
    }

    @Test
    public void fullAttentionUsesGlobalKvHeadsWhenAttentionKEqualsV() {
        Gemma4Config config = config(true);
        assertEquals(2, config.getLayerKeyValueProjectionHeads("full_attention"));
        assertEquals(16, config.getLayerKeyValueProjectionLength("full_attention"));
    }

    @Test
    public void proportionalRopeLeavesNonRotaryTailAsIdentity() {
        Gemma4Config config = config(false);
        float[][] freqs = config.ropeFreqsByLayerType.get("full_attention");
        int halfDim = config.getLayerHeadDim("full_attention") / 2;
        int pos1 = halfDim;
        /*
        original asserts ARM
                assertTrue(Math.abs(freqs[pos1][1]) > 0.0f || Math.abs(freqs[pos1][0] - 1.0f) > 0.0f);
        assertEquals(1.0f, freqs[pos1 + 1][0], 1.0e-6f);
        assertEquals(0.0f, freqs[pos1 + 1][1], 1.0e-6f);
        assertEquals(1.0f, freqs[pos1 + 2][0], 1.0e-6f);
        assertEquals(0.0f, freqs[pos1 + 2][1], 1.0e-6f);
        assertEquals(1.0f, freqs[pos1 + 3][0], 1.0e-6f);
        assertEquals(0.0f, freqs[pos1 + 3][1], 1.0e-6f);
         */
        // headDim=8, partial_rotary_factor=0.25 -> only first pair rotates, remaining pairs stay identity
        assertTrue(Math.abs(freqs[pos1][1]) > 0.0f || Math.abs(freqs[pos1][0] - 1.0f) > 0.0f);
        assertEquals(1.0f, freqs[pos1 + 1][0], 1.0e-3f);
        assertEquals(0.0f, freqs[pos1 + 1][1], .04);
        assertEquals(1.0f, freqs[pos1 + 2][0], 1.0e-3f);
        assertEquals(0.0f, freqs[pos1 + 2][1], 1.0e-3f);
        assertEquals(1.0f, freqs[pos1 + 3][0], 1.0e-3f);
        assertEquals(0.0f, freqs[pos1 + 3][1], 1.0e-3f);
    }

    @Test
    public void defaultRopeRotatesAcrossAllPairs() {
        Gemma4Config config = config(false);
        float[][] freqs = config.ropeFreqsByLayerType.get("sliding_attention");
        int halfDim = config.getLayerHeadDim("sliding_attention") / 2;
        int pos1SecondPair = halfDim + 1;
        // position=1, second pair should also rotate for default rope
        assertTrue(Math.abs(freqs[pos1SecondPair][1]) > 0.0f || Math.abs(freqs[pos1SecondPair][0] - 1.0f) > 0.0f);
    }

    @Test
    public void sharedKvUsesLastNonSharedLayerOfSameType() {
        Gemma4Config config = e2bLikeConfig();
        assertTrue(config.storesSharedKvState(13));
        assertTrue(config.storesSharedKvState(14));
        assertEquals(13, config.getSharedKvSourceLayer(15));
        assertEquals(13, config.getSharedKvSourceLayer(18));
        assertEquals(14, config.getSharedKvSourceLayer(19));
        assertEquals(14, config.getSharedKvSourceLayer(34));
    }

    private static Gemma4Config config(boolean attentionKEqV) {
        Map<String, Object> textConfig = new LinkedHashMap<>();
        textConfig.put("max_position_embeddings", 32);
        textConfig.put("hidden_size", 16);
        textConfig.put("intermediate_size", 32);
        textConfig.put("num_attention_heads", 2);
        textConfig.put("num_key_value_heads", 1);
        textConfig.put("num_global_key_value_heads", 2);
        textConfig.put("num_hidden_layers", 2);
        textConfig.put("rms_norm_eps", 1.0e-6);
        textConfig.put("vocab_size", 64);
        textConfig.put("bos_token_id", 2);
        textConfig.put("eos_token_id", List.of(1));
        textConfig.put("hidden_activation", "gelu_pytorch_tanh");
        textConfig.put("head_dim", 4);
        textConfig.put("global_head_dim", 8);
        textConfig.put("sliding_window", 16);
        textConfig.put("attention_k_eq_v", attentionKEqV);
        textConfig.put("layer_types", List.of("sliding_attention", "full_attention"));
        textConfig.put("rope_parameters", Map.of(
                "sliding_attention", Map.of("rope_theta", 10000.0),
                "full_attention", Map.of("rope_theta", 1000000.0)
        ));
        return new Gemma4Config(textConfig, List.of("Gemma4ForConditionalGeneration"), List.of(1));
    }

    private static Gemma4Config e2bLikeConfig() {
        Map<String, Object> textConfig = new LinkedHashMap<>();
        textConfig.put("max_position_embeddings", 32);
        textConfig.put("hidden_size", 16);
        textConfig.put("intermediate_size", 32);
        textConfig.put("num_attention_heads", 2);
        textConfig.put("num_key_value_heads", 1);
        textConfig.put("num_hidden_layers", 35);
        textConfig.put("num_kv_shared_layers", 20);
        textConfig.put("rms_norm_eps", 1.0e-6);
        textConfig.put("vocab_size", 64);
        textConfig.put("bos_token_id", 2);
        textConfig.put("eos_token_id", List.of(1));
        textConfig.put("hidden_activation", "gelu_pytorch_tanh");
        textConfig.put("head_dim", 4);
        textConfig.put("sliding_window", 16);
        textConfig.put("layer_types", List.of(
                "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"
        ));
        textConfig.put("rope_parameters", Map.of(
                "sliding_attention", Map.of("rope_theta", 10000.0),
                "full_attention", Map.of("rope_theta", 1000000.0, "rope_type", "proportional", "partial_rotary_factor", 0.25)
        ));
        return new Gemma4Config(textConfig, List.of("Gemma4ForConditionalGeneration"), List.of(1));
    }
}
