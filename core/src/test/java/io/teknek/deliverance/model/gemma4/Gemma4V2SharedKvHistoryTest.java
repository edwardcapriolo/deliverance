package io.teknek.deliverance.model.gemma4;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.JsonUtils;
import io.teknek.deliverance.grace.PreTrainedTokenizer;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.tensorparallel.SingleRankTensorParallelCollectives;
import io.teknek.deliverance.model.tensorparallel.StaticTensorParallelContext;
import io.teknek.deliverance.safetensors.DefaultWeightLoader;
import io.teknek.deliverance.safetensors.SafeTensorWriter;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.kv.KvCacheSession;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;
import io.teknek.deliverance.toolcallparser.DefaultToolCallParser;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ForkJoinPool;

import static org.junit.jupiter.api.Assertions.assertEquals;

class Gemma4V2SharedKvHistoryTest {
    @TempDir
    Path tempDir;

    @Test
    void sharedConsumerReadsFullSourceKvHistoryAcrossPrefillAndDecode() throws Exception {
        Path modelDir = writeTinySharedKvCheckpoint(tempDir.resolve("gemma4-shared-kv"));
        MetricRegistry metrics = new MetricRegistry();
        WrappedForkJoinPool pool = new WrappedForkJoinPool(new ForkJoinPool(2));
        List<float[]> finalHiddenRows = new ArrayList<>();
        try (DefaultWeightLoader weights = new DefaultWeightLoader(modelDir.toFile());
             WrappedForkJoinPool ignoredPool = pool;
             Gemma4Model model = new Gemma4Model(
                     AbstractModel.InferenceType.FULL_GENERATION,
                     JsonUtils.om.readValue(modelDir.resolve("config.json").toFile(), Gemma4Config.class),
                     weights,
                     org.mockito.Mockito.mock(PreTrainedTokenizer.class),
                     DType.F32, DType.F32, Optional.empty(), metrics,
                     new KvBufferCacheSettings(true).withBlockSize(2), new DefaultToolCallParser(), pool,
                      new StaticTensorParallelContext(0, 1), new SingleRankTensorParallelCollectives(),
                      Optional.empty(), new Lighter(metrics, pool.getUnderlying()))) {
            model.init();
            model.setLayerDebugHook(event -> {
                if (event.layerIndex() == 1 && "layer_output".equals(event.stage())) {
                    finalHiddenRows.add(lastRow(event.hiddenStates()));
                }
            });

            assertEquals(0, ((Gemma4Config) model.getConfig()).getSharedKvSourceLayer(1));
            try (KvCacheSession session = model.newKvCacheSession()) {
                try (TensorRef ignored = model.batchForwardRef(new int[]{0, 1}, 0, session)) { }
                float[] cachedDecode2;
                float[] cachedDecode3;
                try (TensorRef decode2 = model.forwardRef(2, 2, session, Optional.empty());
                     TensorRef decode3 = model.forwardRef(3, 3, session, Optional.empty())) {
                    cachedDecode2 = lastRow(decode2);
                    cachedDecode3 = lastRow(decode3);
                }

                assertEquals(4, session.length());
                for (int row = 0; row < 4; row++) {
                    try (AbstractTensor key = session.keyRowCopy(0, row);
                         AbstractTensor value = session.valueRowCopy(0, row)) {
                        assertEquals(1, key.shape().first());
                        assertEquals(1, value.shape().first());
                    }
                }

                try (KvCacheSession replay012 = model.newKvCacheSession();
                     TensorRef replay012Output = model.batchForwardRef(new int[]{0, 1, 2}, 0, replay012);
                     KvCacheSession replay0123 = model.newKvCacheSession();
                     TensorRef replay0123Output = model.batchForwardRef(new int[]{0, 1, 2, 3}, 0, replay0123)) {
                    assertClose(lastRow(replay012Output), cachedDecode2, 1.0e-5f,
                            "cached decode 2 must match cold replay [0,1,2]");
                    assertClose(lastRow(replay0123Output), cachedDecode3, 1.0e-5f,
                            "cached decode 3 must match cold replay [0,1,2,3]");
                }

                assertClose(finalHiddenRows.get(1), finalHiddenRows.get(3), 1.0e-5f,
                        "shared consumer hidden state for decode 2 must match cold replay");
                assertClose(finalHiddenRows.get(2), finalHiddenRows.get(4), 1.0e-5f,
                        "shared consumer hidden state for decode 3 must match cold replay");

                try (KvCacheSession first = model.newKvCacheSession();
                     KvCacheSession second = model.newKvCacheSession()) {
                    try (TensorRef ignored = model.batchForwardRef(new int[]{0, 1}, 0, first)) { }
                    try (TensorRef ignored = model.batchForwardRef(new int[]{2, 3}, 0, second)) { }
                    float[] firstDecode;
                    try (TensorRef output = model.forwardRef(2, 2, first, Optional.empty())) {
                        firstDecode = lastRow(output);
                    }
                    float[] secondDecode;
                    try (TensorRef output = model.forwardRef(0, 2, second, Optional.empty())) {
                        secondDecode = lastRow(output);
                    }
                    try (KvCacheSession firstCold = model.newKvCacheSession();
                         TensorRef firstOutput = model.batchForwardRef(new int[]{0, 1, 2}, 0, firstCold);
                         KvCacheSession secondCold = model.newKvCacheSession();
                         TensorRef secondOutput = model.batchForwardRef(new int[]{2, 3, 0}, 0, secondCold)) {
                        assertClose(lastRow(firstOutput), firstDecode, 1.0e-5f,
                                "interleaved first session must match cold replay");
                        assertClose(lastRow(secondOutput), secondDecode, 1.0e-5f,
                                "interleaved second session must match cold replay");
                    }
                }
            }
        }
    }

    private float[] lastRow(AbstractTensor tensor) {
        float[] values = new float[tensor.shape().last()];
        int row = tensor.shape().first() - 1;
        for (int column = 0; column < values.length; column++) values[column] = tensor.get(row, column);
        return values;
    }

    private float[] lastRow(TensorRef tensor) {
        float[] values = new float[tensor.shape().last()];
        int row = tensor.shape().first() - 1;
        for (int column = 0; column < values.length; column++) values[column] = tensor.get(row, column);
        return values;
    }

    private void assertClose(float[] expected, float[] actual, float tolerance, String message) {
        assertEquals(expected.length, actual.length, message + " length");
        for (int column = 0; column < expected.length; column++) {
            assertEquals(expected[column], actual[column], tolerance, message + " column " + column);
        }
    }

    private Path writeTinySharedKvCheckpoint(Path directory) throws Exception {
        Files.createDirectories(directory);
        Map<String, Object> text = new LinkedHashMap<>();
        text.put("max_position_embeddings", 8);
        text.put("hidden_size", 4);
        text.put("intermediate_size", 4);
        text.put("num_attention_heads", 1);
        text.put("num_key_value_heads", 1);
        text.put("num_hidden_layers", 2);
        text.put("rms_norm_eps", 1.0e-6f);
        text.put("vocab_size", 4);
        text.put("bos_token_id", 0);
        text.put("eos_token_id", 3);
        text.put("hidden_activation", "gelu_pytorch_tanh");
        text.put("head_dim", 4);
        text.put("layer_types", List.of("full_attention", "full_attention"));
        text.put("num_kv_shared_layers", 1);
        // Full attention omits v_proj in HF when V aliases K.
        text.put("attention_k_eq_v", true);
        text.put("enable_moe_block", false);
        text.put("use_double_wide_mlp", false);
        text.put("rope_parameters", Map.of("full_attention", Map.of("rope_type", "default", "rope_theta", 10000.0)));
        Map<String, Object> config = new LinkedHashMap<>();
        config.put("model_type", "gemma4");
        config.put("architectures", List.of("Gemma4ForCausalLM"));
        config.put("text_config", text);
        config.put("eos_token_id", 3);
        JsonUtils.om.writeValue(directory.resolve("config.json").toFile(), config);

        Map<String, AbstractTensor> tensors = new LinkedHashMap<>();
        tensors.put("model.language_model.embed_tokens.weight", embedding());
        tensors.put("model.language_model.norm.weight", ones());
        for (int layer = 0; layer < 2; layer++) {
            String prefix = "model.language_model.layers." + layer + ".";
            tensors.put(prefix + "input_layernorm.weight", ones());
            tensors.put(prefix + "post_attention_layernorm.weight", ones());
            tensors.put(prefix + "pre_feedforward_layernorm.weight", ones());
            tensors.put(prefix + "post_feedforward_layernorm.weight", ones());
            tensors.put(prefix + "self_attn.q_proj.weight", zeros(4, 4));
            tensors.put(prefix + "self_attn.q_norm.weight", ones());
            tensors.put(prefix + "self_attn.o_proj.weight", layer == 1 ? identity() : zeros(4, 4));
            tensors.put(prefix + "mlp.gate_proj.weight", zeros(4, 4));
            tensors.put(prefix + "mlp.up_proj.weight", zeros(4, 4));
            tensors.put(prefix + "mlp.down_proj.weight", zeros(4, 4));
            // HF shared consumers have no K/V tensors of their own.  The source
            // layer remains fully materialized so replay also covers its history.
            if (layer == 0) {
                tensors.put(prefix + "self_attn.k_proj.weight", zeros(4, 4));
                tensors.put(prefix + "self_attn.k_norm.weight", ones());
            }
        }
        SafeTensorWriter.writeModel(directory, Map.of("format", "pt"), tensors);
        tensors.values().forEach(AbstractTensor::close);
        return directory;
    }

    private FloatBufferTensor embedding() {
        FloatBufferTensor tensor = new FloatBufferTensor(4, 4);
        for (int row = 0; row < 4; row++) {
            for (int column = 0; column < 4; column++) {
                tensor.set(row == column ? row + 1.0f : 0.0f, row, column);
            }
        }
        return tensor;
    }

    private FloatBufferTensor identity() {
        FloatBufferTensor tensor = zeros(4, 4);
        for (int i = 0; i < 4; i++) tensor.set(1.0f, i, i);
        return tensor;
    }

    private FloatBufferTensor ones() {
        return filled(1, 4, 1.0f);
    }

    private FloatBufferTensor zeros(int rows, int columns) {
        return filled(rows, columns, 0.0f);
    }

    private FloatBufferTensor filled(int rows, int columns, float value) {
        FloatBufferTensor tensor = new FloatBufferTensor(TensorShape.of(rows, columns));
        tensor.getMemorySegment().fill((byte) 0);
        if (value != 0.0f) {
            for (int row = 0; row < rows; row++) {
                for (int column = 0; column < columns; column++) tensor.set(value, row, column);
            }
        }
        return tensor;
    }
}
