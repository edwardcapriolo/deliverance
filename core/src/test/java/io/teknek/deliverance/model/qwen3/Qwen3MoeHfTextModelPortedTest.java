package io.teknek.deliverance.model.qwen3;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.JsonUtils;
import io.teknek.deliverance.grace.PreTrainedTokenizer;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.model.CausalLanguageModel;
import io.teknek.deliverance.model.DefaultCausalLanguageModel;
import io.teknek.deliverance.model.ModelSupport;
import io.teknek.deliverance.model.tensorparallel.SingleRankTensorParallelCollectives;
import io.teknek.deliverance.model.tensorparallel.StaticTensorParallelContext;
import io.teknek.deliverance.safetensors.DefaultWeightLoader;
import io.teknek.deliverance.safetensors.SafeTensorWriter;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.KvBufferCache;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.NaiveTensorOperations;
import io.teknek.deliverance.toolcallparser.DefaultToolCallParser;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.io.TempDir;
import org.mockito.Mockito;

import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Ports feasible tiny-checkpoint mechanics from HF tests/models/qwen3_moe/test_modeling_qwen3_moe.py. */
public class Qwen3MoeHfTextModelPortedTest {
    @TempDir
    Path tempDir;

    @Test
    public void modelSupportDetectsQwen3MoeConfig() throws Exception {
        Path modelDir = writeTinyCheckpoint(tempDir.resolve("qwen3-moe-detect"), tinyConfig(), 1234);

        assertEquals(Qwen3MoeConfig.class, ModelSupport.detectModel(modelDir.resolve("config.json").toFile()).getConfigClass());
    }

    @Test
    public void tinyModelForwardReturnsExpectedShape() {
        Path modelDir = writeTinyCheckpoint(tempDir.resolve("qwen3-moe-shape"), tinyConfig(), 2234);
        try (Qwen3MoeModel model = loadTinyModel(modelDir);
             KvBufferCache.KvBuffer kv = model.newKvBuffer();
             AbstractTensor output = model.batchForward(new int[]{3, 4, 5, 6}, 0, kv)) {
            assertEquals(4, output.shape().first());
            assertEquals(model.getConfig().embeddingLength, output.shape().last());
            assertFinite(output);
        }
    }

    @Test
    public void tinyModelForwardIsDeterministicForSameInput() {
        Path modelDir = writeTinyCheckpoint(tempDir.resolve("qwen3-moe-deterministic"), tinyConfig(), 3234);
        int[] tokens = new int[]{3, 4, 5, 6};
        try (Qwen3MoeModel model = loadTinyModel(modelDir);
             KvBufferCache.KvBuffer firstKv = model.newKvBuffer();
             KvBufferCache.KvBuffer secondKv = model.newKvBuffer();
             AbstractTensor first = model.batchForward(tokens, 0, firstKv);
             AbstractTensor second = model.batchForward(tokens, 0, secondKv)) {
            assertEquals(0.0f, drift(first, second).maxAbs(), "same model/input should be deterministic");
        }
    }

    @Test
    public void differentSeededCheckpointsChangeForwardOutput() {
        Path firstModelDir = writeTinyCheckpoint(tempDir.resolve("qwen3-moe-first"), tinyConfig(), 3334);
        Path secondModelDir = writeTinyCheckpoint(tempDir.resolve("qwen3-moe-second"), tinyConfig(), 4334);
        int[] tokens = new int[]{3, 4, 5, 6};
        try (Qwen3MoeModel firstModel = loadTinyModel(firstModelDir);
             Qwen3MoeModel secondModel = loadTinyModel(secondModelDir);
             KvBufferCache.KvBuffer firstKv = firstModel.newKvBuffer();
             KvBufferCache.KvBuffer secondKv = secondModel.newKvBuffer();
             AbstractTensor first = firstModel.batchForward(tokens, 0, firstKv);
             AbstractTensor second = secondModel.batchForward(tokens, 0, secondKv)) {
            assertTrue(drift(first, second).maxAbs() > 1.0e-6f, "different checkpoints should change output");
        }
    }

    @Test
    public void tinyModelDecodeMatchesColdReplayForFixedContinuation() {
        Path modelDir = writeTinyCheckpoint(tempDir.resolve("qwen3-moe-decode"), tinyConfig(), 4234);
        int[] prompt = new int[]{3, 4, 5, 6};
        int[] continuation = new int[]{7, 8};
        try (Qwen3MoeModel model = loadTinyModel(modelDir);
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

    @Test
    public void sparseAndDenseLayerTensorNamesLoad() {
        Qwen3MoeConfig config = tinyConfigWithDenseLayer();
        Path modelDir = writeTinyCheckpoint(tempDir.resolve("qwen3-moe-dense-and-sparse"), config, 5234);
        try (Qwen3MoeModel model = loadTinyModel(modelDir);
             KvBufferCache.KvBuffer kv = model.newKvBuffer();
             AbstractTensor output = model.batchForward(new int[]{3, 4, 5}, 0, kv)) {
            assertFinite(output);
        }
    }

    @Disabled("Quantize-on-demand coverage for Qwen3 MoE expert tensors; enable when validating Q4 conversion path.")
    @Test
    public void tinyModelQuantizeOnDemandLoadsAndRuns() throws Exception {
        Path cacheDir = tempDir.resolve("cache");
        Path sourceDir = cacheDir.resolve("acme_qwen3-moe-tiny");
        writeTinyCheckpoint(sourceDir, tinyConfig(), 6234);
        ModelFetcher source = new ModelFetcher("acme", "qwen3-moe-tiny");
        source.setBaseDir(cacheDir);

        try (CausalLanguageModel model = AutoModelForCausaLm.newBuilder(source)
                .withDownload(false)
                .withQuantizeOnDemand(DType.Q4, "acme", "qwen3-moe-tiny-q4")
                .build();
             KvBufferCache.KvBuffer kv = ((DefaultCausalLanguageModel) model).localTransformerModel().newKvBuffer();
             AbstractTensor output = ((DefaultCausalLanguageModel) model).localTransformerModel()
                     .batchForward(new int[]{3, 4, 5, 6}, 0, kv)) {
            assertEquals(4, output.shape().first());
            assertEquals(tinyConfig().embeddingLength, output.shape().last());
            assertFinite(output);
        }
    }

    static Qwen3MoeConfig tinyConfig() {
        return new Qwen3MoeConfig(64, 16, 32, 2, 1, 3, 1.0e-6f, 32, null, 1,
                ActivationFunction.Type.SILU, 10_000.0, Map.of("rope_type", "default", "rope_theta", 10_000.0),
                8, false, null, null, null, 0.0f, List.of("Qwen3MoeForCausalLM"),
                1, 8, 2, 4, true, false, 0.001f, List.of());
    }

    static Qwen3MoeConfig tinyConfigWithDenseLayer() {
        return new Qwen3MoeConfig(64, 16, 32, 2, 1, 3, 1.0e-6f, 32, null, 1,
                ActivationFunction.Type.SILU, 10_000.0, Map.of("rope_type", "default", "rope_theta", 10_000.0),
                8, false, null, null, null, 0.0f, List.of("Qwen3MoeForCausalLM"),
                1, 8, 2, 4, true, false, 0.001f, List.of(1));
    }

    static Qwen3MoeModel loadTinyModel(Path modelDir) {
        MetricRegistry metrics = new MetricRegistry();
        TensorAllocator allocator = new ArrayQueueTensorAllocator(metrics);
        WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores());
        return new Qwen3MoeModel(AbstractModel.InferenceType.FULL_GENERATION, configFromFile(modelDir),
                new DefaultWeightLoader(modelDir.toFile()), Mockito.mock(PreTrainedTokenizer.class), DType.F32, DType.I8,
                Optional.empty(), new ConfigurableTensorProvider(new NaiveTensorOperations()), metrics, allocator,
                new KvBufferCacheSettings(true), new DefaultToolCallParser(), pool,
                new StaticTensorParallelContext(0, 1), new SingleRankTensorParallelCollectives(), Optional.empty());
    }

    static Path writeTinyCheckpoint(Path dir, Qwen3MoeConfig config, int seed) {
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
                if (config.sparseLayer(i)) {
                    tensors.put(layer + "mlp.gate.weight", matrix(config.numExperts, config.embeddingLength, seed++));
                    tensors.put(layer + "mlp.experts.gate_up_proj",
                            tensor3(config.numExperts, 2 * config.moeIntermediateSize, config.embeddingLength, seed++));
                    tensors.put(layer + "mlp.experts.down_proj",
                            tensor3(config.numExperts, config.embeddingLength, config.moeIntermediateSize, seed++));
                } else {
                    tensors.put(layer + "mlp.gate_proj.weight", matrix(config.hiddenLength, config.embeddingLength, seed++));
                    tensors.put(layer + "mlp.up_proj.weight", matrix(config.hiddenLength, config.embeddingLength, seed++));
                    tensors.put(layer + "mlp.down_proj.weight", matrix(config.embeddingLength, config.hiddenLength, seed++));
                }
            }
            SafeTensorWriter.writeModel(dir, Map.of("format", "pt"), tensors, 1 << 28);
            tensors.values().forEach(AbstractTensor::close);
            return dir;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    static Map<String, Object> tinyConfigJson(Qwen3MoeConfig config) {
        Map<String, Object> json = new LinkedHashMap<>();
        json.put("model_type", "qwen3_moe");
        json.put("architectures", List.of("Qwen3MoeForCausalLM"));
        json.put("max_position_embeddings", config.contextLength);
        json.put("hidden_size", config.embeddingLength);
        json.put("intermediate_size", config.hiddenLength);
        json.put("num_attention_heads", config.numberOfHeads);
        json.put("num_key_value_heads", config.numberOfKeyValueHeads);
        json.put("num_hidden_layers", config.numberOfLayers);
        json.put("rms_norm_eps", config.layerNormEps);
        json.put("vocab_size", config.vocabularySize);
        json.put("eos_token_id", config.eosTokens.getFirst());
        json.put("hidden_act", "silu");
        json.put("head_dim", config.headSize);
        json.put("rope_parameters", Map.of("rope_type", "default", "rope_theta", 10_000.0));
        json.put("use_sliding_window", config.useSlidingWindow);
        json.put("attention_dropout", config.attentionDropout);
        json.put("decoder_sparse_step", config.decoderSparseStep);
        json.put("moe_intermediate_size", config.moeIntermediateSize);
        json.put("num_experts_per_tok", config.numExpertsPerToken);
        json.put("num_experts", config.numExperts);
        json.put("norm_topk_prob", config.normTopkProb);
        json.put("output_router_logits", config.outputRouterLogits);
        json.put("router_aux_loss_coef", config.routerAuxLossCoef);
        json.put("mlp_only_layers", config.mlpOnlyLayers);
        return json;
    }

    static Qwen3MoeConfig configFromFile(Path modelDir) {
        try {
            return JsonUtils.om.readValue(modelDir.resolve("config.json").toFile(), Qwen3MoeConfig.class);
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

    static FloatBufferTensor tensor3(int first, int second, int third, int seed) {
        FloatBufferTensor tensor = new FloatBufferTensor(first, second, third);
        for (int i = 0; i < first; i++) {
            for (int j = 0; j < second; j++) {
                for (int k = 0; k < third; k++) {
                    tensor.set(((i * 43 + j * 19 + k * 11 + seed) % 23 - 11) / 12.0f, i, j, k);
                }
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

    private static AbstractTensor coldReplay(Qwen3MoeModel model, int[] prompt, int[] continuation, int continuationLength) {
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
