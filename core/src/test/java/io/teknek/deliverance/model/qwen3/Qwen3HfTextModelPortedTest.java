package io.teknek.deliverance.model.qwen3;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.JsonUtils;
import io.teknek.deliverance.grace.PreTrainedTokenizer;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.hf.HfConfigTesterMixinPort;
import io.teknek.deliverance.model.hf.HfGenerationTesterMixinPort;
import io.teknek.deliverance.model.hf.HfModelTesterMixinPort;
import io.teknek.deliverance.model.hf.HfUnsupportedMixinPort;
import io.teknek.deliverance.model.tensorparallel.SingleRankTensorParallelCollectives;
import io.teknek.deliverance.model.tensorparallel.StaticTensorParallelContext;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.safetensors.DefaultWeightLoader;
import io.teknek.deliverance.safetensors.SafeTensorWriter;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.KvBufferCache;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.NaiveTensorOperations;
import io.teknek.deliverance.toolcallparser.DefaultToolCallParser;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.mockito.Mockito;

import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Ports the feasible Qwen3 text-model tests from Hugging Face.
 *
 * <p>Sources:</p>
 * <ul>
 *     <li>/ai-code/transformers/tests/models/qwen3/test_modeling_qwen3.py</li>
 *     <li>/ai-code/transformers/tests/causal_lm_tester.py</li>
 *     <li>/ai-code/transformers/tests/test_modeling_common.py</li>
 * </ul>
 */
public class Qwen3HfTextModelPortedTest implements
        HfConfigTesterMixinPort,
        HfModelTesterMixinPort,
        HfGenerationTesterMixinPort,
        HfUnsupportedMixinPort {

    @TempDir
    Path tempDir;

    @Override
    public Path hfTestTempDir() {
        return tempDir;
    }

    @Override
    public Path writeTinyCheckpoint(String name, int seed) {
        return writeTinyCheckpoint(tempDir.resolve(name), tinyConfig(), seed);
    }

    @Override
    public Qwen3Model loadTinyModel(Path modelDir) {
        return loadTinyQwen3Model(modelDir);
    }

    @Override
    public Qwen3Config loadTinyConfig(Path modelDir) {
        return configFromFile(modelDir);
    }

    @Override
    public Config roundTripConfig(Config config) throws Exception {
        Qwen3Config qwen3 = (Qwen3Config) config;
        String json = JsonUtils.om.writeValueAsString(tinyConfigJson(qwen3));
        return JsonUtils.om.readValue(json, Qwen3Config.class);
    }

    @Override
    public int[] hfSampleTokenIds() {
        return new int[]{3, 4, 5, 6};
    }

    @Override
    public AbstractTensor makeInputsEmbeds(int rows, int embeddingLength, int seed) {
        return matrix(rows, embeddingLength, seed);
    }

    @Override
    public void assertModelSpecificConfigRoundTrip(Config expected, Config actual) {
        Qwen3Config first = (Qwen3Config) expected;
        Qwen3Config second = (Qwen3Config) actual;
        assertEquals(first.layerTypes, second.layerTypes);
        assertEquals(first.slidingWindow, second.slidingWindow);
        assertEquals(first.maxWindowLayers, second.maxWindowLayers);
        assertEquals(first.attentionDropout, second.attentionDropout);
    }

    @Test
    public void hfQwen3ModelTesterConfigShape() {
        Qwen3Config config = tinyConfig();

        assertEquals(4, config.numberOfLayers);
        assertEquals(2, config.numberOfHeads);
        assertEquals(1, config.numberOfKeyValueHeads);
        assertEquals(List.of("full_attention", "full_attention", "full_attention", "full_attention"), config.layerTypes);
    }

    @Test
    public void tinyModelForwardReturnsExpectedShape() {
        Path modelDir = writeTinyCheckpoint(tempDir.resolve("qwen3-tiny-shape"), tinyConfig(), 1234);
        try (Qwen3Model model = loadTinyModel(modelDir);
             KvBufferCache.KvBuffer kv = model.newKvBuffer();
             AbstractTensor output = model.batchForward(new int[]{3, 4, 5, 6}, 0, kv)) {
            assertEquals(4, output.shape().first());
            assertEquals(model.getConfig().embeddingLength, output.shape().last());
            assertFinite(output);
        }
    }

    @Test
    public void tinyModelForwardIsDeterministicForSameInput() {
        Path modelDir = writeTinyCheckpoint(tempDir.resolve("qwen3-tiny-deterministic"), tinyConfig(), 2234);
        int[] tokens = new int[]{3, 4, 5, 6};
        try (Qwen3Model model = loadTinyModel(modelDir);
             KvBufferCache.KvBuffer firstKv = model.newKvBuffer();
             KvBufferCache.KvBuffer secondKv = model.newKvBuffer();
             AbstractTensor first = model.batchForward(tokens, 0, firstKv);
             AbstractTensor second = model.batchForward(tokens, 0, secondKv)) {
            assertEquals(0.0f, drift(first, second).maxAbs(), "same model/input should be deterministic");
        }
    }

    @Test
    public void tinyModelDecodeMatchesColdReplayForFixedContinuation() {
        Path modelDir = writeTinyCheckpoint(tempDir.resolve("qwen3-tiny-decode"), tinyConfig(), 3234);
        int[] prompt = new int[]{3, 4, 5, 6};
        int[] continuation = new int[]{7, 8, 9};
        try (Qwen3Model model = loadTinyModel(modelDir);
             KvBufferCache.KvBuffer decodeKv = model.newKvBuffer();
             AbstractTensor promptOutput = model.batchForward(prompt, 0, decodeKv)) {
            promptOutput.close();
            for (int i = 0; i < continuation.length; i++) {
                try (AbstractTensor decode = model.forward(continuation[i], prompt.length + i, decodeKv);
                     AbstractTensor replay = coldReplay(model, prompt, continuation, i + 1)) {
                    assertTrue(driftLastBatchRow(replay, decode).maxAbs() < 1.0e-4f,
                            "decode should match cold replay at step " + i);
                }
            }
        }
    }

    @Override
    @Test
    @Disabled("HF Qwen3 supports inputs_embeds; Deliverance generic path permits it, but no dedicated parity assertion is ported yet")
    public void hfGenerationInputsEmbedsUnsupportedWhenModelRequiresInputIds() {
    }

    @Test
    @Disabled("HF slow integration test loads Qwen/Qwen3-0.6B-Base; Deliverance keeps real model checks as separate ITs")
    public void testModel600mLogits() {
    }

    @Test
    @Disabled("HF slow integration test loads Qwen/Qwen3-0.6B-Base; Deliverance keeps real model checks as separate ITs")
    public void testModel600mGeneration() {
    }

    @Test
    @Disabled("HF test targets FlashAttention/sliding-window long prompt path; Deliverance does not expose FA2")
    public void testModel600mLongPrompt() {
    }

    @Test
    @Disabled("HF test targets SDPA/sliding-window long prompt path; Deliverance has no SDPA backend")
    public void testModel600mLongPromptSdpa() {
    }

    static Qwen3Config tinyConfig() {
        return new Qwen3Config(32, 16, 32, 2, 1, 4, 1.0e-6f, 64, null, 2,
                ActivationFunction.Type.SILU, 10_000.0, Map.of("rope_type", "default", "rope_theta", 10_000.0),
                8, false, null, 28, null, 0.0f, List.of("Qwen3ForCausalLM"));
    }

    static Map<String, Object> tinyConfigJson(Qwen3Config config) {
        return new LinkedHashMap<>(Map.ofEntries(
                Map.entry("model_type", "qwen3"),
                Map.entry("architectures", List.of("Qwen3ForCausalLM")),
                Map.entry("max_position_embeddings", config.contextLength),
                Map.entry("hidden_size", config.embeddingLength),
                Map.entry("intermediate_size", config.hiddenLength),
                Map.entry("num_attention_heads", config.numberOfHeads),
                Map.entry("num_key_value_heads", config.numberOfKeyValueHeads),
                Map.entry("num_hidden_layers", config.numberOfLayers),
                Map.entry("rms_norm_eps", config.layerNormEps),
                Map.entry("vocab_size", config.vocabularySize),
                Map.entry("bos_token_id", config.bosToken),
                Map.entry("eos_token_id", config.eosTokens.getFirst()),
                Map.entry("hidden_act", "silu"),
                Map.entry("head_dim", config.headSize),
                Map.entry("rope_parameters", Map.of("rope_type", "default", "rope_theta", 10_000.0)),
                Map.entry("use_sliding_window", config.useSlidingWindow),
                Map.entry("attention_dropout", config.attentionDropout)
        ));
    }

    static Qwen3Model loadTinyQwen3Model(Path modelDir) {
        MetricRegistry metrics = new MetricRegistry();
        TensorAllocator allocator = new ArrayQueueTensorAllocator(metrics);
        WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores());
        return new Qwen3Model(AbstractModel.InferenceType.FULL_GENERATION, configFromFile(modelDir),
                new DefaultWeightLoader(modelDir.toFile()), Mockito.mock(PreTrainedTokenizer.class), DType.F32, DType.I8,
                Optional.empty(), new ConfigurableTensorProvider(new NaiveTensorOperations()), metrics, allocator,
                new KvBufferCacheSettings(true), new DefaultToolCallParser(), pool,
                new StaticTensorParallelContext(0, 1), new SingleRankTensorParallelCollectives(), Optional.empty());
    }

    static Path writeTinyCheckpoint(Path dir, Qwen3Config config, int seed) {
        try {
            java.nio.file.Files.createDirectories(dir);
            JsonUtils.om.writeValue(dir.resolve("config.json").toFile(), tinyConfigJson(config));
            Map<String, AbstractTensor> tensors = new LinkedHashMap<>();
            tensors.put("model.embed_tokens.weight", matrix(config.vocabularySize, config.embeddingLength, seed++));
            tensors.put("model.norm.weight", ones(1, config.embeddingLength));
            for (int i = 0; i < config.numberOfLayers; i++) {
                String layer = "model.layers." + i + ".";
                tensors.put(layer + "input_layernorm.weight", ones(1, config.embeddingLength));
                tensors.put(layer + "post_attention_layernorm.weight", ones(1, config.embeddingLength));
                tensors.put(layer + "self_attn.q_proj.weight", matrix(config.attentionLength, config.embeddingLength, seed++));
                tensors.put(layer + "self_attn.k_proj.weight", matrix(config.kvLength, config.embeddingLength, seed++));
                tensors.put(layer + "self_attn.v_proj.weight", matrix(config.kvLength, config.embeddingLength, seed++));
                tensors.put(layer + "self_attn.o_proj.weight", matrix(config.embeddingLength, config.attentionLength, seed++));
                tensors.put(layer + "self_attn.q_norm.weight", ones(1, config.headSize));
                tensors.put(layer + "self_attn.k_norm.weight", ones(1, config.headSize));
                tensors.put(layer + "mlp.gate_proj.weight", matrix(config.hiddenLength, config.embeddingLength, seed++));
                tensors.put(layer + "mlp.up_proj.weight", matrix(config.hiddenLength, config.embeddingLength, seed++));
                tensors.put(layer + "mlp.down_proj.weight", matrix(config.embeddingLength, config.hiddenLength, seed++));
            }
            SafeTensorWriter.writeModel(dir, Map.of("format", "pt"), tensors, 1 << 28);
            tensors.values().forEach(AbstractTensor::close);
            return dir;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    static Qwen3Config configFromFile(Path modelDir) {
        try {
            return JsonUtils.om.readValue(modelDir.resolve("config.json").toFile(), Qwen3Config.class);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    static FloatBufferTensor matrix(int rows, int cols, int seed) {
        FloatBufferTensor tensor = new FloatBufferTensor(rows, cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                tensor.set(((row * 13 + col * 7 + seed) % 17 - 8) / 8.0f, row, col);
            }
        }
        return tensor;
    }

    static FloatBufferTensor ones(int rows, int cols) {
        FloatBufferTensor tensor = new FloatBufferTensor(rows, cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                tensor.set(1.0f, row, col);
            }
        }
        return tensor;
    }

    private static AbstractTensor coldReplay(Qwen3Model model, int[] prompt, int[] continuation, int continuationLength) {
        int[] tokens = java.util.Arrays.copyOf(prompt, prompt.length + continuationLength);
        System.arraycopy(continuation, 0, tokens, prompt.length, continuationLength);
        return model.batchForward(tokens, 0);
    }

    private static void assertFinite(AbstractTensor tensor) {
        for (int row = 0; row < tensor.shape().first(); row++) {
            for (int col = 0; col < tensor.shape().last(); col++) {
                assertTrue(Float.isFinite(tensor.get(row, col)));
            }
        }
    }

    private static Drift driftLastBatchRow(AbstractTensor batchOutput, AbstractTensor singleRowOutput) {
        int row = batchOutput.shape().first() - 1;
        float max = 0.0f;
        double total = 0.0;
        for (int col = 0; col < batchOutput.shape().last(); col++) {
            float diff = Math.abs(batchOutput.get(row, col) - singleRowOutput.get(0, col));
            max = Math.max(max, diff);
            total += diff;
        }
        return new Drift(max, total / batchOutput.shape().last());
    }

    private static Drift drift(AbstractTensor first, AbstractTensor second) {
        float max = 0.0f;
        double total = 0.0;
        int count = 0;
        for (int row = 0; row < first.shape().first(); row++) {
            for (int col = 0; col < first.shape().last(); col++) {
                float diff = Math.abs(first.get(row, col) - second.get(row, col));
                max = Math.max(max, diff);
                total += diff;
                count++;
            }
        }
        return new Drift(max, total / count);
    }

    private record Drift(float maxAbs, double meanAbs) {
    }
}
