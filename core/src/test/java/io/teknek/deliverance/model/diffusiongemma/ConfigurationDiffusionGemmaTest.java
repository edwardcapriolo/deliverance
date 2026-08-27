package io.teknek.deliverance.model.diffusiongemma;

import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.JsonUtils;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import com.fasterxml.jackson.databind.exc.ValueInstantiationException;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static io.teknek.deliverance.model.diffusiongemma.DiffusionGemmaTextConfig.BidirectionalAttention.ALL;
import static io.teknek.deliverance.model.diffusiongemma.DiffusionGemmaTextConfig.BidirectionalAttention.VISION;
import static io.teknek.deliverance.model.diffusiongemma.DiffusionGemmaTextConfig.LayerType.FULL_ATTENTION;
import static io.teknek.deliverance.model.diffusiongemma.DiffusionGemmaTextConfig.LayerType.SLIDING_ATTENTION;
import static io.teknek.deliverance.model.diffusiongemma.DiffusionGemmaTextConfig.RopeType.DEFAULT;
import static io.teknek.deliverance.model.diffusiongemma.DiffusionGemmaTextConfig.RopeType.PROPORTIONAL;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ConfigurationDiffusionGemmaTest {

    @Test
    public void textConfigUsesHuggingFaceDefaults() {
        DiffusionGemmaTextConfig config = new DiffusionGemmaTextConfig();

        assertEquals(262_144, config.vocabSize);
        assertEquals(2304, config.hiddenSize);
        assertEquals(9216, config.intermediateSize);
        assertEquals(30, config.numHiddenLayers);
        assertEquals(8, config.numAttentionHeads);
        assertEquals(4, config.numKeyValueHeads);
        assertEquals(256, config.headDim);
        assertEquals(ActivationFunction.Type.GELU_PYTORCH_TANH, config.hiddenActivation);
        assertEquals(131_072, config.maxPositionEmbeddings);
        assertEquals(1.0e-6f, config.rmsNormEps, 0.0f);
        assertEquals(0, config.padTokenId);
        assertEquals(1, config.eosTokenId);
        assertEquals(2, config.bosTokenId);
        assertTrue(config.tieWordEmbeddings);
        assertFalse(config.attentionBias);
        assertEquals(0.0f, config.attentionDropout, 0.0f);
        assertEquals(512, config.slidingWindow);
        assertEquals(30.0f, config.finalLogitSoftcapping, 0.0f);
        assertNull(config.useBidirectionalAttention);
        assertNull(config.numExperts);
        assertNull(config.topKExperts);
        assertNull(config.moeIntermediateSize);
        assertTrue(config.causal);
    }

    @Test
    public void defaultsLayerTypesWithFiveSlidingThenFullPatternAndForcesFinalLayerFull() {
        DiffusionGemmaTextConfig config = tinyTextConfig(null, null, null);

        assertEquals(List.of(SLIDING_ATTENTION, SLIDING_ATTENTION, SLIDING_ATTENTION, SLIDING_ATTENTION,
                SLIDING_ATTENTION, FULL_ATTENTION, SLIDING_ATTENTION, FULL_ATTENTION), config.layerTypes);
    }

    @Test
    public void explicitLayerTypesStillForceFinalLayerFullAttention() {
        DiffusionGemmaTextConfig config = tinyTextConfig(List.of("sliding_attention", "sliding_attention"), null, null);

        assertEquals(List.of(SLIDING_ATTENTION, FULL_ATTENTION), config.layerTypes);
    }

    @Test
    public void useBidirectionalAttentionAllDisablesCausalAndAdjustsSlidingWindow() {
        DiffusionGemmaTextConfig config = tinyTextConfig(null, ALL, null);

        assertFalse(config.causal);
        assertEquals(257, config.slidingWindow);
    }

    @Test
    public void useBidirectionalAttentionVisionPreservesCausalTextBehavior() {
        DiffusionGemmaTextConfig config = tinyTextConfig(null, VISION, null);

        assertTrue(config.causal);
        assertEquals(512, config.slidingWindow);
    }

    @Test
    public void defaultsRopeParametersByLayerType() {
        DiffusionGemmaTextConfig config = new DiffusionGemmaTextConfig();

        assertEquals(DEFAULT, config.ropeParameters.get(SLIDING_ATTENTION).ropeType());
        assertEquals(10_000.0, config.ropeParameters.get(SLIDING_ATTENTION).ropeTheta(), 0.0);
        assertNull(config.ropeParameters.get(SLIDING_ATTENTION).partialRotaryFactor());
        assertEquals(PROPORTIONAL, config.ropeParameters.get(FULL_ATTENTION).ropeType());
        assertEquals(1_000_000.0, config.ropeParameters.get(FULL_ATTENTION).ropeTheta(), 0.0);
        assertEquals(0.25, config.ropeParameters.get(FULL_ATTENTION).partialRotaryFactor(), 0.0);
    }

    @Test
    public void parsesExplicitRopeParametersWithEnums() {
        DiffusionGemmaTextConfig config = tinyTextConfig(null, null, Map.of(
                "sliding_attention", Map.of("rope_type", "default", "rope_theta", 123.0),
                "full_attention", Map.of("rope_type", "proportional", "partial_rotary_factor", 0.5, "rope_theta", 456.0)
        ));

        assertEquals(DEFAULT, config.ropeParameters.get(SLIDING_ATTENTION).ropeType());
        assertEquals(123.0, config.ropeParameters.get(SLIDING_ATTENTION).ropeTheta(), 0.0);
        assertEquals(PROPORTIONAL, config.ropeParameters.get(FULL_ATTENTION).ropeType());
        assertEquals(456.0, config.ropeParameters.get(FULL_ATTENTION).ropeTheta(), 0.0);
        assertEquals(0.5, config.ropeParameters.get(FULL_ATTENTION).partialRotaryFactor(), 0.0);
    }

    @Test
    public void topLevelConfigDefaultsTextAndSpecialTokens() {
        DiffusionGemmaConfig config = new DiffusionGemmaConfig();

        assertEquals(256, config.canvasLength);
        assertEquals(255_999, config.boiTokenId);
        assertEquals(258_882, config.eoiTokenId);
        assertEquals(258_880, config.imageTokenId);
        assertEquals(0.02f, config.initializerRange, 0.0f);
        assertTrue(config.tieWordEmbeddings);
        assertEquals(262_144, config.textConfig.vocabSize);
        assertNull(config.visionConfig);
    }

    @Test
    public void topLevelConfigDefaultsVisionConfigModelType() {
        DiffusionGemmaConfig config = new DiffusionGemmaConfig(null, Map.of("hidden_size", 16), null,
                null, null, null, null, null);

        assertEquals("gemma4_vision", config.visionConfig.get("model_type"));
        assertEquals(16, config.visionConfig.get("hidden_size"));
    }

    @Test
    public void parsesSnakeCaseJsonIntoTextAndTopLevelConfig() throws Exception {
        String json = """
                {
                  "canvas_length": 16,
                  "boi_token_id": 5,
                  "eoi_token_id": 6,
                  "image_token_id": 4,
                  "tie_word_embeddings": false,
                  "vision_config": {"hidden_size": 16},
                  "text_config": {
                    "vocab_size": 128,
                    "hidden_size": 32,
                    "intermediate_size": 64,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 2,
                    "head_dim": 16,
                    "hidden_activation": "gelu_pytorch_tanh",
                    "max_position_embeddings": 512,
                    "rms_norm_eps": 0.000001,
                    "pad_token_id": 0,
                    "eos_token_id": [1, 2],
                    "bos_token_id": 3,
                    "layer_types": ["sliding_attention", "sliding_attention"],
                    "use_bidirectional_attention": "all",
                    "num_global_key_value_heads": 2,
                    "global_head_dim": 16,
                    "num_experts": 4,
                    "top_k_experts": 2,
                    "moe_intermediate_size": 8
                  }
                }
                """;

        DiffusionGemmaConfig config = JsonUtils.om.readValue(json, DiffusionGemmaConfig.class);

        assertEquals(16, config.canvasLength);
        assertEquals(5, config.boiTokenId);
        assertEquals(6, config.eoiTokenId);
        assertEquals(4, config.imageTokenId);
        assertFalse(config.tieWordEmbeddings);
        assertEquals("gemma4_vision", config.visionConfig.get("model_type"));
        assertEquals(128, config.textConfig.vocabSize);
        assertEquals(List.of(1, 2), config.textConfig.eosTokenId);
        assertEquals(List.of(SLIDING_ATTENTION, FULL_ATTENTION), config.textConfig.layerTypes);
        assertEquals(ALL, config.textConfig.useBidirectionalAttention);
        assertFalse(config.textConfig.causal);
        assertEquals(4, config.textConfig.numExperts);
        assertEquals(2, config.textConfig.topKExperts);
        assertEquals(8, config.textConfig.moeIntermediateSize);
    }

    @Test
    public void preservesScalarAndListEosTokenForms() throws Exception {
        DiffusionGemmaTextConfig scalar = JsonUtils.om.readValue("{\"eos_token_id\": 7}",
                DiffusionGemmaTextConfig.class);
        DiffusionGemmaTextConfig list = JsonUtils.om.readValue("{\"eos_token_id\": [7, 8]}",
                DiffusionGemmaTextConfig.class);

        assertEquals(7, scalar.eosTokenId);
        assertEquals(List.of(7, 8), list.eosTokenId);
    }

    @Test
    public void explicitTopLevelValuesOverrideDefaults() {
        DiffusionGemmaConfig config = new DiffusionGemmaConfig(new DiffusionGemmaTextConfig(), null,
                11, 12, 13, 0.04f, false, 32);

        assertEquals(32, config.canvasLength);
        assertEquals(11, config.boiTokenId);
        assertEquals(12, config.eoiTokenId);
        assertEquals(13, config.imageTokenId);
        assertEquals(0.04f, config.initializerRange, 0.0f);
        assertFalse(config.tieWordEmbeddings);
    }

    @Test
    public void parsesExplicitMoeFields() {
        DiffusionGemmaTextConfig config = tinyTextConfig(null, null, null);

        assertEquals(4, config.numExperts);
        assertEquals(2, config.topKExperts);
        assertEquals(8, config.moeIntermediateSize);
        assertEquals(2, config.numGlobalKeyValueHeads);
        assertEquals(16, config.globalHeadDim);
    }

    @Test
    public void invalidLayerTypeFailsFast() {
        assertThrows(IllegalArgumentException.class,
                () -> tinyTextConfig(List.of("banana_attention"), null, null));
    }

    @Test
    public void invalidBidirectionalAttentionJsonFailsFast() {
        assertThrows(ValueInstantiationException.class,
                () -> JsonUtils.om.readValue("{\"use_bidirectional_attention\": \"banana\"}",
                        DiffusionGemmaTextConfig.class));
    }

    @Test
    public void invalidRopeTypeFailsFast() {
        Map<String, Map<String, Object>> rope = new LinkedHashMap<>();
        rope.put("sliding_attention", Map.of("rope_type", "banana", "rope_theta", 123.0));

        assertThrows(IllegalArgumentException.class, () -> tinyTextConfig(null, null, rope));
    }

    @Disabled("Requires DiffusionGemma model implementation and tiny checkpoint writer")
    @Test
    public void modelCanBeConstructedFromConfig() {
    }

    @Disabled("Requires vision config/model implementation")
    @Test
    public void visionConfigMappingCreatesGemma4VisionConfig() {
    }

    private static DiffusionGemmaTextConfig tinyTextConfig(List<String> layerTypes,
            DiffusionGemmaTextConfig.BidirectionalAttention bidirectionalAttention,
            Map<String, Map<String, Object>> ropeParameters) {
        return new DiffusionGemmaTextConfig(
                128,
                32,
                32,
                8,
                2,
                2,
                16,
                "gelu_pytorch_tanh",
                512,
                0.02f,
                1.0e-6f,
                0,
                1,
                2,
                true,
                ropeParameters,
                false,
                0.0f,
                512,
                layerTypes,
                30.0f,
                bidirectionalAttention,
                2,
                16,
                4,
                2,
                8
        );
    }
}
