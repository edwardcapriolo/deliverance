package io.teknek.deliverance.model.gemma4;

import org.junit.jupiter.api.Test;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
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
        Map<String, Object> textConfig = baseConfig(false);
        textConfig.put("rope_parameters", Map.of(
                "sliding_attention", Map.of("rope_theta", 10000.0),
                "full_attention", Map.of("rope_theta", 1000000.0, "rope_type", "proportional", "partial_rotary_factor", 0.25)
        ));
        Gemma4Config config = new Gemma4Config(textConfig, List.of("Gemma4ForConditionalGeneration"), List.of(1));
        float[] freqs = config.ropeInvFreqsByLayerType.get("full_attention");
        int halfDim = config.getLayerHeadDim("full_attention") / 2;
        // headDim=8, partial_rotary_factor=0.25 -> only first pair rotates, remaining pairs stay identity
        assertTrue(freqs[0] > 0.0f);
        for (int i = 1; i < halfDim; i++) {
            assertEquals(0.0f, freqs[i], 1.0e-6f);
        }
    }

    @Test
    public void proportionalRopeDoesNotForceRotatedPairWhenPartialFactorIsZero() {
        Map<String, Object> textConfig = baseConfig(false);
        textConfig.put("rope_parameters", Map.of(
                "sliding_attention", Map.of("rope_theta", 10000.0),
                "full_attention", Map.of("rope_theta", 1000000.0, "rope_type", "proportional", "partial_rotary_factor", 0.0)
        ));
        Gemma4Config config = new Gemma4Config(textConfig, List.of("Gemma4ForConditionalGeneration"), List.of(1));
        float[] freqs = config.ropeInvFreqsByLayerType.get("full_attention");
        int halfDim = config.getLayerHeadDim("full_attention") / 2;

        for (int i = 0; i < halfDim; i++) {
            assertEquals(0.0f, freqs[i], 1.0e-6f);
        }
    }

    @Test
    public void defaultRopeRotatesAcrossAllPairs() {
        Gemma4Config config = config(false);
        float[] freqs = config.ropeInvFreqsByLayerType.get("sliding_attention");
        int halfDim = config.getLayerHeadDim("sliding_attention") / 2;
        for (int i = 0; i < halfDim; i++) {
            assertTrue(freqs[i] > 0.0f);
        }
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

    @Test
    public void scalarIntermediateSizePreservesDoubleWideSharedLayerRule() {
        Map<String, Object> textConfig = e2bLikeTextConfig();
        textConfig.put("use_double_wide_mlp", true);
        Gemma4Config config = new Gemma4Config(textConfig, List.of("Gemma4ForConditionalGeneration"), List.of(1));

        assertEquals(32, config.getLayerHiddenLength(0));
        assertEquals(64, config.getLayerHiddenLength(15));
    }

    @Test
    public void defaultsLayerTypesAndForcesLastLayerFullAttention() {
        Map<String, Object> textConfig = e2bLikeTextConfig();
        textConfig.remove("layer_types");
        Gemma4Config config = new Gemma4Config(textConfig, List.of("Gemma4ForConditionalGeneration"), List.of(1));

        assertEquals("sliding_attention", config.layerTypes.get(0));
        assertEquals("full_attention", config.layerTypes.get(5));
        assertEquals("full_attention", config.layerTypes.getLast());
    }

    @Test
    public void defaultsRopeParametersByLayerType() {
        Map<String, Object> textConfig = e2bLikeTextConfig();
        textConfig.remove("rope_parameters");
        Gemma4Config config = new Gemma4Config(textConfig, List.of("Gemma4ForConditionalGeneration"), List.of(1));

        assertEquals("default", config.ropeParametersByLayerType.get("sliding_attention").get("rope_type"));
        assertEquals("proportional", config.ropeParametersByLayerType.get("full_attention").get("rope_type"));
    }

    @Test
    public void parsesExpertIntermediateSizeAlias() {
        Map<String, Object> textConfig = e2bLikeTextConfig();
        textConfig.put("expert_intermediate_size", 1234);
        Gemma4Config config = new Gemma4Config(textConfig, List.of("Gemma4ForConditionalGeneration"), List.of(1));

        assertEquals(1234, config.moeIntermediateSize);
    }

    @Test
    public void finalSoftcapDoesNotBecomeAttentionSoftcap() {
        Map<String, Object> textConfig = e2bLikeTextConfig();
        textConfig.put("final_logit_softcapping", 30.0);
        Gemma4Config config = new Gemma4Config(textConfig, List.of("Gemma4ForConditionalGeneration"), List.of(1));

        assertEquals(30.0f, config.finalLogitSoftCapping);
        assertNull(config.attnLogitSoftCapping);
    }

    private static Gemma4Config config(boolean attentionKEqV) {
        return new Gemma4Config(baseConfig(attentionKEqV), List.of("Gemma4ForConditionalGeneration"), List.of(1));
    }

    private static Map<String, Object> baseConfig(boolean attentionKEqV) {
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
        return textConfig;
    }

    private static Gemma4Config e2bLikeConfig() {
        return new Gemma4Config(e2bLikeTextConfig(), List.of("Gemma4ForConditionalGeneration"), List.of(1));
    }

    private static Map<String, Object> e2bLikeTextConfig() {
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
        return textConfig;
    }
}
