package io.teknek.deliverance.model.gpt2;

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
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.kv.KvCacheSession;
import io.teknek.deliverance.tensor2.TensorRef;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.NaiveTensorOperations;
import io.teknek.deliverance.toolcallparser.DefaultToolCallParser;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.mockito.Mockito;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ForkJoinPool;

import static org.junit.jupiter.api.Assertions.assertEquals;

class Gpt2TensorRefTest {
    @TempDir
    Path tempDir;

    @Test
    void tinyCheckpointMatchesLegacyPrefillAndCachedDecode() {
        Path checkpoint = writeCheckpoint(tempDir.resolve("gpt2-tiny"));
        int[] prompt = {1, 3, 5};
        try (Gpt2Model ref = load(checkpoint, true);
             Gpt2Model legacy = load(checkpoint, false);
             KvCacheSession refSession = ref.newKvCacheSession();
             io.teknek.deliverance.tensor.KvBufferCache.KvBuffer legacyCache = legacy.newKvBuffer();
             TensorRef actual = ref.batchForwardRef(prompt, 0, refSession);
             AbstractTensor expected = legacy.batchForward(prompt, 0, legacyCache)) {
            assertTensorClose(expected, actual);
            try (AbstractTensor expectedDecode = legacy.forward(7, prompt.length, legacyCache);
                 TensorRef actualDecode = ref.forwardRef(7, prompt.length, refSession)) {
                assertTensorClose(expectedDecode, actualDecode);
            }
            legacyCache.close();
        }
    }

    private Gpt2Model load(Path checkpoint, boolean tensorRef) {
        MetricRegistry metrics = new MetricRegistry();
        TensorAllocator allocator = new ArrayQueueTensorAllocator(metrics);
        WrappedForkJoinPool pool = new WrappedForkJoinPool(new ForkJoinPool(2));
        Gpt2Model model = new Gpt2Model(AbstractModel.InferenceType.FULL_GENERATION,
                config(), new DefaultWeightLoader(checkpoint.toFile()), Mockito.mock(PreTrainedTokenizer.class),
                DType.F32, DType.F32, Optional.empty(), new ConfigurableTensorProvider(
                new NaiveTensorOperations()), metrics, allocator, new KvBufferCacheSettings(true),
                new DefaultToolCallParser(), pool, new StaticTensorParallelContext(0, 1),
                new SingleRankTensorParallelCollectives(), Optional.empty()) {
            @Override
            protected boolean usesTensorRefExecution() {
                return tensorRef;
            }

            @Override
            public boolean usesKvCache2Generation() {
                return tensorRef;
            }
        };
        model.init();
        return model;
    }

    private Gpt2Config config() {
        return new Gpt2Config(8, 8, 2, 1, 1.0e-5f, 16, 1, 2);
    }

    private Path writeCheckpoint(Path dir) {
        try {
            Files.createDirectories(dir);
            Map<String, Object> config = new LinkedHashMap<>();
            config.put("model_type", "gpt2");
            config.put("n_ctx", 8);
            config.put("n_embd", 8);
            config.put("n_head", 2);
            config.put("n_layer", 1);
            config.put("layer_norm_epsilon", 1.0e-5);
            config.put("vocab_size", 16);
            config.put("bos_token_id", 1);
            config.put("eos_token_id", 2);
            JsonUtils.om.writeValue(dir.resolve("config.json").toFile(), config);
            Map<String, AbstractTensor> tensors = new LinkedHashMap<>();
            tensors.put("wte.weight", matrix(16, 8, 1));
            tensors.put("wpe.weight", matrix(8, 8, 2));
            tensors.put("ln_f.weight", ones(1, 8));
            tensors.put("ln_f.bias", zeros(1, 8));
            tensors.put("h.0.ln_1.weight", ones(1, 8));
            tensors.put("h.0.ln_1.bias", zeros(1, 8));
            tensors.put("h.0.ln_2.weight", ones(1, 8));
            tensors.put("h.0.ln_2.bias", zeros(1, 8));
            tensors.put("h.0.attn.c_attn.weight", matrix(8, 24, 3));
            tensors.put("h.0.attn.c_attn.bias", matrix(1, 24, 4));
            tensors.put("h.0.attn.c_proj.weight", matrix(8, 8, 5));
            tensors.put("h.0.attn.c_proj.bias", matrix(1, 8, 6));
            tensors.put("h.0.mlp.c_fc.weight", matrix(8, 32, 7));
            tensors.put("h.0.mlp.c_fc.bias", matrix(1, 32, 8));
            tensors.put("h.0.mlp.c_proj.weight", matrix(32, 8, 9));
            tensors.put("h.0.mlp.c_proj.bias", matrix(1, 8, 10));
            SafeTensorWriter.writeModel(dir, Map.of("format", "pt"), tensors, 1 << 20);
            tensors.values().forEach(AbstractTensor::close);
            return dir;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private FloatBufferTensor matrix(int rows, int columns, int seed) {
        FloatBufferTensor tensor = new FloatBufferTensor(rows, columns);
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                tensor.set(((row * 3 + column * 5 + seed) % 17 - 8) / 80.0f, row, column);
            }
        }
        return tensor;
    }

    private FloatBufferTensor ones(int rows, int columns) {
        FloatBufferTensor tensor = new FloatBufferTensor(rows, columns);
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) tensor.set(1.0f, row, column);
        }
        return tensor;
    }

    private FloatBufferTensor zeros(int rows, int columns) {
        return new FloatBufferTensor(rows, columns);
    }

    private void assertTensorClose(AbstractTensor expected, TensorRef actual) {
        assertEquals(expected.shape(), actual.shape());
        for (int row = 0; row < expected.shape().first(); row++) {
            for (int column = 0; column < expected.shape().last(); column++) {
                assertEquals(expected.get(row, column), actual.get(row, column), 1.0e-4f,
                        "row=" + row + " column=" + column);
            }
        }
    }
}
