package io.teknek.deliverance.model.nemotronlabsdiffusion;

import io.teknek.deliverance.JsonUtils;
import io.teknek.deliverance.math.ActivationFunction;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

class NemotronLabsDiffusionConfigTest {
    @Test
    void parsesBaseConfigValuesFromUpstreamJson() throws Exception {
        String json = """
                {
                  "architectures": ["NemotronLabsDiffusionModel"],
                  "attention_bias": false,
                  "attention_dropout": 0.0,
                  "attn_implementation": "sdpa",
                  "block_size": 32,
                  "bos_token_id": 1,
                  "dlm_loss_weight": null,
                  "dlm_paradigm": "bidirectional",
                  "dp_varying_mask_ratio": false,
                  "eos_token_id": 2,
                  "head_dim": 128,
                  "hidden_act": "silu",
                  "hidden_size": 3072,
                  "intermediate_size": 9216,
                  "mask_token_id": 100,
                  "max_position_embeddings": 4096,
                  "mlp_bias": false,
                  "model_type": "nemotron_labs_diffusion",
                  "num_attention_heads": 32,
                  "num_hidden_layers": 26,
                  "num_key_value_heads": 8,
                  "rms_norm_eps": 1e-05,
                  "rope_parameters": {
                    "beta_fast": 32.0,
                    "beta_slow": 1.0,
                    "factor": 0.25,
                    "llama_4_scaling_beta": 0.1,
                    "mscale": 1.0,
                    "mscale_all_dim": 1.0,
                    "original_max_position_embeddings": 16384,
                    "rope_theta": 1000000.0,
                    "rope_type": "yarn"
                  },
                  "sliding_window": null,
                  "tie_word_embeddings": false,
                  "vocab_size": 131072
                }
                """;

        NemotronLabsDiffusionConfig config = JsonUtils.om.readValue(json, NemotronLabsDiffusionConfig.class);

        assertEquals(4096, config.contextLength);
        assertEquals(3072, config.embeddingLength);
        assertEquals(9216, config.hiddenLength);
        assertEquals(26, config.numberOfLayers);
        assertEquals(32, config.numberOfHeads);
        assertEquals(8, config.numberOfKeyValueHeads);
        assertEquals(128, config.headSize);
        assertEquals(4096, config.attentionLength);
        assertEquals(1024, config.kvLength);
        assertEquals(4, config.headGroupSize);
        assertEquals(ActivationFunction.Type.SILU, config.activationFunction);
        assertEquals(131072, config.vocabularySize);
        assertEquals(1, config.bosToken);
        assertEquals(2, config.eosTokens.getFirst());
        assertEquals(100, config.maskTokenId);
        assertEquals(32, config.blockSize);
        assertEquals("bidirectional", config.dlmParadigm);
        assertEquals("sdpa", config.attnImplementation);
        assertFalse(config.attentionBias);
        assertFalse(config.mlpBias);
        assertEquals("yarn", config.ropeParameters.get("rope_type"));
        assertEquals(0.25d, ((Number) config.ropeParameters.get("factor")).doubleValue());
        assertEquals(0.1d, ((Number) config.ropeParameters.get("llama_4_scaling_beta")).doubleValue());
    }
}
