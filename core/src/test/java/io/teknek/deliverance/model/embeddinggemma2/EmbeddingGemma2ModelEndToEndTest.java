package io.teknek.deliverance.model.embeddinggemma2;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.JsonUtils;
import io.teknek.deliverance.embedding.PoolingType;
import io.teknek.deliverance.generator.EmbedInput;
import io.teknek.deliverance.grace.EncodeOptions;
import io.teknek.deliverance.grace.Encoding;
import io.teknek.deliverance.grace.PreTrainedTokenizer;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.tensorparallel.SingleRankTensorParallelCollectives;
import io.teknek.deliverance.model.tensorparallel.StaticTensorParallelContext;
import io.teknek.deliverance.safetensors.DefaultWeightLoader;
import io.teknek.deliverance.safetensors.SafeTensorWriter;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.NaiveTensorOperations;
import io.teknek.deliverance.toolcallparser.DefaultToolCallParser;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ForkJoinPool;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/** Loads a real tiny SafeTensors layout through the model loader and executes text embedding end to end. */
class EmbeddingGemma2ModelEndToEndTest {
    private static final int VOCAB = 8;
    private static final int HIDDEN = 4;
    private static final int INTERMEDIATE = 4;
    private static final int EMBEDDING = 128;

    @Test
    void loadsTinyCheckpointAndReturnsNormalizedTruncatedEmbedding(@TempDir Path directory) throws Exception {
        writeCheckpoint(directory);
        PreTrainedTokenizer tokenizer = mock(PreTrainedTokenizer.class);
        when(tokenizer.encode(anyString(), any(EncodeOptions.class))).thenReturn(
                new Encoding(new int[] {1, 2, 3}, new int[] {1, 1, 1}, new int[] {0, 0, 0}));
        MetricRegistry metrics = new MetricRegistry();
        TensorAllocator allocator = new ArrayQueueTensorAllocator(metrics);
        WrappedForkJoinPool pool = new WrappedForkJoinPool(new ForkJoinPool(1));
        EmbeddingGemma2Config config = JsonUtils.om.readValue(
                directory.resolve("config.json").toFile(), EmbeddingGemma2Config.class);
        try (EmbeddingGemma2Model model = new EmbeddingGemma2Model(AbstractModel.InferenceType.FULL_EMBEDDING,
                config, new DefaultWeightLoader(directory.toFile()), tokenizer, DType.F32, DType.F32,
                Optional.empty(), new ConfigurableTensorProvider(new NaiveTensorOperations()), metrics, allocator,
                new KvBufferCacheSettings(true), new DefaultToolCallParser(), pool,
                new StaticTensorParallelContext(0, 1), new SingleRankTensorParallelCollectives(), Optional.empty())) {
            model.init();
            float[] embedding = model.embed("ignored", new EmbeddingGemma2EmbeddingOptions(null, null, 128, true));
            assertEquals(128, embedding.length);
            double norm = 0.0;
            for (float value : embedding) {
                assertTrue(Float.isFinite(value));
                norm += (double) value * value;
            }
            assertEquals(1.0, Math.sqrt(norm), 1.0e-4);
        }
        pool.close();
    }

    private static void writeCheckpoint(Path directory) throws Exception {
        Files.createDirectories(directory);
        Map<String, Object> text = new LinkedHashMap<>();
        text.put("vocab_size", VOCAB);
        text.put("hidden_size", HIDDEN);
        text.put("intermediate_size", INTERMEDIATE);
        text.put("num_hidden_layers", 1);
        text.put("num_attention_heads", 1);
        text.put("num_key_value_heads", 1);
        text.put("head_dim", HIDDEN);
        text.put("hidden_activation", "gelu_pytorch_tanh");
        text.put("max_position_embeddings", 16);
        text.put("rms_norm_eps", 1.0e-6);
        text.put("pad_token_id", 0);
        text.put("bos_token_id", 2);
        text.put("eos_token_id", 1);
        text.put("sliding_window", 4);
        text.put("embedding_dim", EMBEDDING);
        text.put("hidden_size_per_layer_input", HIDDEN);
        text.put("layer_types", java.util.List.of("full_attention"));
        text.put("rope_parameters", Map.of("full_attention", Map.of("rope_theta", 10_000.0)));
        text.put("per_layer_config", Map.of());
        Map<String, Object> config = new LinkedHashMap<>();
        config.put("model_type", "embedding_gemma2");
        config.put("architectures", java.util.List.of("EmbeddingGemma2Model"));
        config.put("image_token_id", 1);
        config.put("video_token_id", 2);
        config.put("audio_token_id", 3);
        config.put("boi_token_id", 4);
        config.put("eoi_token_id", 5);
        config.put("boa_token_id", 6);
        config.put("eoa_token_index", 7);
        config.put("text_config", text);
        JsonUtils.om.writeValue(directory.resolve("config.json").toFile(), config);

        Map<String, io.teknek.deliverance.tensor.AbstractTensor> tensors = new LinkedHashMap<>();
        tensors.put("embed_tokens.weight", matrix(VOCAB, HIDDEN, 1.0f));
        tensors.put("ple.per_layer_model_projection.weight", matrix(HIDDEN, HIDDEN, 0.02f));
        tensors.put("ple.per_layer_projection_norm.weight", ones(1, HIDDEN));
        tensors.put("norm.weight", ones(1, HIDDEN));
        tensors.put("embedding_projection.weight", matrix(EMBEDDING, HIDDEN, 0.01f));
        String base = "layers.0.";
        tensors.put(base + "input_layernorm.weight", ones(1, HIDDEN));
        tensors.put(base + "self_attn.q_proj.weight", identity(HIDDEN));
        tensors.put(base + "self_attn.k_proj.weight", identity(HIDDEN));
        tensors.put(base + "self_attn.v_proj.weight", identity(HIDDEN));
        tensors.put(base + "self_attn.o_proj.weight", identity(HIDDEN));
        tensors.put(base + "self_attn.q_norm.weight", ones(1, HIDDEN));
        tensors.put(base + "self_attn.k_norm.weight", ones(1, HIDDEN));
        tensors.put(base + "post_attention_layernorm.weight", ones(1, HIDDEN));
        tensors.put(base + "pre_feedforward_layernorm.weight", ones(1, HIDDEN));
        tensors.put(base + "mlp.gate_proj.weight", identity(HIDDEN));
        tensors.put(base + "mlp.up_proj.weight", identity(HIDDEN));
        tensors.put(base + "mlp.down_proj.weight", identity(HIDDEN));
        tensors.put(base + "post_feedforward_layernorm.weight", ones(1, HIDDEN));
        tensors.put(base + "ple_block.per_layer_input_gate.weight", identity(HIDDEN));
        tensors.put(base + "ple_block.per_layer_projection.weight", identity(HIDDEN));
        tensors.put(base + "ple_block.post_per_layer_input_norm.weight", ones(1, HIDDEN));
        SafeTensorWriter.writeModel(directory, Map.of("format", "pt"), tensors, 1 << 20);
    }

    private static FloatBufferTensor matrix(int rows, int columns, float value) {
        FloatBufferTensor tensor = new FloatBufferTensor(rows, columns);
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                tensor.set(value * (row + column + 1), row, column);
            }
        }
        return tensor;
    }

    private static FloatBufferTensor identity(int size) {
        FloatBufferTensor tensor = new FloatBufferTensor(size, size);
        for (int i = 0; i < size; i++) tensor.set(1.0f, i, i);
        return tensor;
    }

    private static FloatBufferTensor ones(int rows, int columns) {
        return matrix(rows, columns, 1.0f);
    }
}
