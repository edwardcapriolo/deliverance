package io.teknek.deliverance.model.embeddinggemma2;

import io.teknek.deliverance.model.ModelSupport;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** Verifies model-type detection for the text-capable EmbeddingGemma 2 loader. */
class EmbeddingGemma2ModelRegistrationTest {
    @Test
    void detectsEmbeddingGemma2ModelType(@TempDir Path directory) throws Exception {
        Path config = directory.resolve("config.json");
        Files.writeString(config, """
                {
                  "model_type": "embedding_gemma2",
                  "architectures": ["EmbeddingGemma2Model"],
                  "image_token_id": 1,
                  "video_token_id": 2,
                  "audio_token_id": 3,
                  "boi_token_id": 4,
                  "eoi_token_id": 5,
                  "boa_token_id": 6,
                  "eoa_token_index": 7,
                  "text_config": {
                    "vocab_size": 32,
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 4,
                    "hidden_activation": "gelu_pytorch_tanh",
                    "max_position_embeddings": 32,
                    "rms_norm_eps": 0.000001,
                    "pad_token_id": 0,
                    "bos_token_id": 2,
                    "eos_token_id": 1,
                    "sliding_window": 4,
                    "embedding_dim": 8,
                    "hidden_size_per_layer_input": 8,
                    "layer_types": ["full_attention"],
                    "rope_parameters": {"full_attention": {"rope_theta": 10000.0}},
                    "per_layer_config": {}
                  }
                }
                """);

        assertEquals(EmbeddingGemma2Config.class,
                ModelSupport.detectModel(config.toFile()).getConfigClass());
        var modelType = ModelSupport.detectModel(config.toFile());
        assertEquals(EmbeddingGemma2Model.class, modelType.getModelClass());
        assertEquals(false, modelType.useMajorityDType());
    }
}
