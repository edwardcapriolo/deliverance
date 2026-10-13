package io.teknek.deliverance.model.embeddinggemma2;

import io.teknek.deliverance.JsonUtils;
import com.fasterxml.jackson.databind.exc.ValueInstantiationException;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Tag;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class EmbeddingGemma2ConfigTest {
    /**
     * Port of {@code EmbeddingGemma2TextModelTest.test_config} from
     * {@code transformers/tests/models/embedding_gemma2/test_modeling_embedding_gemma2.py:159-160}.
     *
     * <p>The Python test delegates to Transformers' inherited ConfigTester.
     * This method ports the model-specific tiny configuration contract that
     * ConfigTester feeds into those common checks.</p>
     */
    @Test
    void testConfig() {
        Map<String, Object> tiny = Map.ofEntries(
                Map.entry("vocab_size", 99),
                Map.entry("hidden_size", 32),
                Map.entry("intermediate_size", 37),
                Map.entry("num_hidden_layers", 2),
                Map.entry("num_attention_heads", 4),
                Map.entry("num_key_value_heads", 2),
                Map.entry("head_dim", 8),
                Map.entry("max_position_embeddings", 512),
                Map.entry("sliding_window", 4),
                Map.entry("hidden_size_per_layer_input", 16),
                Map.entry("embedding_dim", 24),
                Map.entry("pad_token_id", 0),
                Map.entry("bos_token_id", 2),
                Map.entry("eos_token_id", 1),
                Map.entry("rms_norm_eps", 0.000001),
                Map.entry("hidden_activation", "gelu_pytorch_tanh"),
                Map.entry("layer_types", List.of("sliding_attention", "full_attention")),
                Map.entry("rope_parameters", Map.of(
                        "sliding_attention", Map.of("rope_theta", 10000.0),
                        "full_attention", Map.of("rope_theta", 1000000.0))),
                Map.entry("per_layer_config", Map.of(
                        "1", Map.of("head_dim", 16, "num_key_value_heads", 2)))
        );

        EmbeddingGemma2TextConfig config = new EmbeddingGemma2TextConfig(tiny);

        assertEquals(99, config.vocabSize);
        assertEquals(32, config.hiddenSize);
        assertEquals(37, config.intermediateSize);
        assertEquals(2, config.numHiddenLayers);
        assertEquals(4, config.numAttentionHeads);
        assertEquals(2, config.numKeyValueHeads);
        assertEquals(8, config.headDimFor("sliding_attention"));
        assertEquals(16, config.headDimFor("1"));
        assertEquals(24, config.embeddingDim);
        assertEquals(List.of("sliding_attention", "full_attention"), config.layerTypes);
        assertEquals(List.of(1), config.eosTokenId);
    }

    @Test
    @Tag("integration")
    void downloadsAllCurrentMetadataAndReadsTheRealConfig() throws Exception {
        Path cache = Files.createTempDirectory("embeddinggemma2-fetch");
        ModelFetcher fetcher = new ModelFetcher("google", "embeddinggemma-2");
        fetcher.setBaseDir(cache);

        File modelDirectory = fetcher.maybeDownload(ModelFetcher.FetchPolicy.TOKENIZER_ONLY);

        EmbeddingGemma2Config config = JsonUtils.om.readValue(
                modelDirectory.toPath().resolve("config.json").toFile(), EmbeddingGemma2Config.class);
        assertEquals("embedding_gemma2", config.modelType);
        assertEquals(768, config.textConfig.embeddingDim);
        assertEquals(24, config.textConfig.layerTypes.size());
        assertEquals(4, config.textConfig.perLayerConfig.size());
        assertEquals(4, config.textConfig.layerTypes.stream()
                .filter("full_attention"::equals)
                .count());
        assertEquals(512, config.textConfig.headDimFor(5));
        assertEquals(1, config.textConfig.keyValueHeadsFor(5));
        for (String metadataFile : List.of(
                "1_Pooling/config.json",
                "2_Normalize/config.json",
                "README.md",
                "chat_template.jinja",
                "config.json",
                "config_sentence_transformers.json",
                "modules.json",
                "preprocessor_config.json",
                "processor_config.json",
                "sentence_bert_config.json",
                "tokenizer.json",
                "tokenizer.model",
                "tokenizer_config.json")) {
            assertTrue(Files.isRegularFile(modelDirectory.toPath().resolve(metadataFile)),
                    "missing fetched metadata file " + metadataFile);
        }
        assertFalse(Files.exists(modelDirectory.toPath().resolve("model.safetensors")));
    }

    @Test
    void readsRealModelConfigurationShape() throws Exception {
        String json = """
                {
                  "architectures": ["EmbeddingGemma2Model"],
                  "audio_token_id": 258881,
                  "boa_token_id": 256000,
                  "boi_token_id": 255999,
                  "eoa_token_index": 258883,
                  "eoi_token_id": 258882,
                  "image_token_id": 258880,
                  "model_type": "embedding_gemma2",
                  "video_token_id": 258884,
                  "text_config": {
                    "embedding_dim": 768,
                    "head_dim": 256,
                    "hidden_activation": "gelu_pytorch_tanh",
                    "hidden_size": 512,
                    "hidden_size_per_layer_input": 512,
                    "intermediate_size": 2048,
                    "layer_types": ["sliding_attention", "full_attention"],
                    "max_position_embeddings": 262144,
                    "num_attention_heads": 4,
                    "num_hidden_layers": 2,
                    "num_key_value_heads": 2,
                    "pad_token_id": 0,
                    "bos_token_id": 2,
                    "eos_token_id": 1,
                    "rms_norm_eps": 0.000001,
                    "sliding_window": 512,
                    "vocab_size": 262144,
                    "rope_parameters": {
                      "sliding_attention": {"rope_theta": 10000.0},
                      "full_attention": {"rope_theta": 1000000.0}
                    },
                    "per_layer_config": {
                      "full_attention": {"head_dim": 512, "num_key_value_heads": 1}
                    }
                  },
                  "vision_config": {"model_type": "gemma4_vision"},
                  "audio_config": {"model_type": "gemma4_audio"}
                }
                """;

        EmbeddingGemma2Config config = JsonUtils.om.readValue(json, EmbeddingGemma2Config.class);

        assertEquals("embedding_gemma2", config.modelType);
        assertEquals("EmbeddingGemma2Model", config.architectures.getFirst());
        assertEquals(512, config.textConfig.hiddenSize);
        assertEquals(768, config.textConfig.embeddingDim);
        assertEquals(256, config.textConfig.headDimFor("sliding_attention"));
        assertEquals(512, config.textConfig.headDimFor("full_attention"));
        assertEquals(2, config.textConfig.keyValueHeadsFor("sliding_attention"));
        assertEquals(1, config.textConfig.keyValueHeadsFor("full_attention"));
        assertEquals(258880, config.imageTokenId);
        assertEquals(258881, config.audioTokenId);
        assertEquals(258884, config.videoTokenId);
        assertTrue(config.visionConfig.containsKey("model_type"));
        assertTrue(config.audioConfig.containsKey("model_type"));
        assertFalse(config.textConfig.ropeParameters.isEmpty());
    }

    @Test
    void rejectsOtherModelTypes() {
        Map<String, Object> text = Map.ofEntries(
                Map.entry("vocab_size", 10),
                Map.entry("hidden_size", 4),
                Map.entry("intermediate_size", 8),
                Map.entry("num_hidden_layers", 1),
                Map.entry("num_attention_heads", 1),
                Map.entry("num_key_value_heads", 1),
                Map.entry("head_dim", 4),
                Map.entry("hidden_activation", "gelu"),
                Map.entry("max_position_embeddings", 8),
                Map.entry("rms_norm_eps", 0.000001),
                Map.entry("pad_token_id", 0),
                Map.entry("bos_token_id", 2),
                Map.entry("eos_token_id", 1),
                Map.entry("sliding_window", 4),
                Map.entry("embedding_dim", 4),
                Map.entry("hidden_size_per_layer_input", 4),
                Map.entry("layer_types", java.util.List.of("full_attention")),
                Map.entry("rope_parameters", Map.of("full_attention", Map.of("rope_theta", 10000.0))),
                Map.entry("per_layer_config", Map.of())
        );
        Map<String, Object> root = Map.of(
                "model_type", "gemma4",
                "architectures", java.util.List.of("Gemma4Model"),
                "text_config", text,
                "image_token_id", 1,
                "video_token_id", 2,
                "audio_token_id", 3,
                "boi_token_id", 4,
                "eoi_token_id", 5,
                "boa_token_id", 6,
                "eoa_token_index", 7
        );

        org.junit.jupiter.api.Assertions.assertThrows(
                ValueInstantiationException.class,
                () -> JsonUtils.om.treeToValue(JsonUtils.om.valueToTree(root), EmbeddingGemma2Config.class));
    }
}
