package io.teknek.deliverance.model.granitemoehybrid;

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
import io.teknek.deliverance.model.ModelSupport;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.model.tensorparallel.SingleRankTensorParallelCollectives;
import io.teknek.deliverance.model.tensorparallel.StaticTensorParallelContext;
import io.teknek.deliverance.safetensors.DefaultWeightLoader;
import io.teknek.deliverance.safetensors.SafeTensorWriter;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.KvBufferCache;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.TensorInfo;
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
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Ports the initial GraniteMoeHybrid layout checks from HF tests/models/granitemoehybrid/test_modeling_granitemoehybrid.py. */
public class GraniteMoeHybridHfTextModelPortedTest implements
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
    public GraniteMoeHybridModel loadTinyModel(Path modelDir) {
        return loadTinyModelStatic(modelDir);
    }

    @Override
    public GraniteMoeHybridConfig loadTinyConfig(Path modelDir) {
        return configFromFile(modelDir);
    }

    @Override
    public Config roundTripConfig(Config config) throws Exception {
        GraniteMoeHybridConfig granite = (GraniteMoeHybridConfig) config;
        String json = JsonUtils.om.writeValueAsString(tinyConfigJson(granite));
        return JsonUtils.om.readValue(json, GraniteMoeHybridConfig.class);
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
        GraniteMoeHybridConfig first = (GraniteMoeHybridConfig) expected;
        GraniteMoeHybridConfig second = (GraniteMoeHybridConfig) actual;
        assertEquals(first.attentionDropout, second.attentionDropout);
        assertEquals(first.sharedIntermediateSize, second.sharedIntermediateSize);
        assertEquals(first.numLocalExperts, second.numLocalExperts);
        assertEquals(first.numExpertsPerToken, second.numExpertsPerToken);
        assertEquals(first.outputRouterLogits, second.outputRouterLogits);
        assertEquals(first.routerAuxLossCoef, second.routerAuxLossCoef);
        assertEquals(first.layerTypes, second.layerTypes);
        assertEquals(first.positionEmbeddingType, second.positionEmbeddingType);
        assertEquals(first.mambaNHeads, second.mambaNHeads);
        assertEquals(first.mambaNGroups, second.mambaNGroups);
        assertEquals(first.mambaDState, second.mambaDState);
        assertEquals(first.mambaDHead, second.mambaDHead);
        assertEquals(first.mambaDConv, second.mambaDConv);
        assertEquals(first.mambaExpand, second.mambaExpand);
        assertEquals(first.mambaChunkSize, second.mambaChunkSize);
        assertEquals(first.mambaConvBias, second.mambaConvBias);
        assertEquals(first.mambaProjBias, second.mambaProjBias);
        assertEquals(first.logitsScaling, second.logitsScaling);
    }

    @Test
    @Disabled("HF GraniteMoeHybrid attention output tests require exposing attention tensors; Deliverance validates forward/KV behavior instead")
    public void testAttentionOutputs() {
    }

    @Test
    @Disabled("HF GraniteMoeHybrid batching equivalence relies on torch batching APIs; Deliverance has dedicated forward determinism and decode replay checks")
    public void testBatchingEquivalence() {
    }

    @Test
    @Disabled("HF GraniteMoeHybrid left padding compatibility requires public attention_mask/position_ids APIs")
    public void testLeftPaddingCompatibility() {
    }

    @Test
    @Disabled("HF GraniteMoeHybrid flash attention padding tests target FlashAttention-specific code paths")
    public void testFlashAttention2PaddingMatchesPaddingFreeWithPositionIds() {
    }

    @Test
    @Disabled("HF GraniteMoeHybrid flash attention padding tests target FlashAttention-specific code paths")
    public void testFlashAttention2PaddingMatchesPaddingFreeWithPositionIdsAndFaKwargs() {
    }

    @Test
    @Disabled("HF GraniteMoeHybrid flash attention seq_idx tests target FlashAttention-specific code paths")
    public void testFlashAttention2PaddingMatchesPaddingFreeWithPositionIdsSeqIdxAndFaKwargs() {
    }

    @Test
    @Disabled("HF GraniteMoeHybrid integration logits test uses ibm-granite/granite-4.0-h-tiny; Deliverance tracks Antares real-model logits in AntaresFetchIT")
    public void testModelLogits() {
    }

    @Test
    @Disabled("HF GraniteMoeHybrid integration generation test uses ibm-granite/granite-4.0-h-tiny; Deliverance tracks Antares generation smoke in AntaresFetchIT")
    public void testModelGeneration() {
    }

    @Test
    @Disabled("HF GraniteMoeHybrid tokenizer encoding test belongs to tokenizer parity; Deliverance Granite model port focuses on model tensor layout and forward path")
    public void testTokenizerEncodingDigitStrings() {
    }

    @Test
    public void modelSupportDetectsGraniteMoeHybridConfig() throws Exception {
        Path modelDir = writeTinyCheckpoint(tempDir.resolve("granite-moe-hybrid-detect"), tinyConfig(), 1234);

        assertEquals(GraniteMoeHybridConfig.class,
                ModelSupport.detectModel(modelDir.resolve("config.json").toFile()).getConfigClass());
    }

    @Test
    public void realAntaresConfigShapeFormulasMatchCheckpointKeyLayout() {
        GraniteMoeHybridConfig real = realAntaresConfig();

        assertEquals(2048, real.embeddingLength);
        assertEquals(4096, real.hiddenLength);
        assertEquals(40, real.numberOfLayers);
        assertEquals(16, real.numberOfHeads);
        assertEquals(4, real.numberOfKeyValueHeads);
        assertEquals(128, real.headSize);
        assertEquals(2048, real.attentionLength);
        assertEquals(512, real.kvLength);
        assertEquals(4096, real.sharedIntermediateSize);
        assertEquals(0, real.numLocalExperts);
        assertEquals(0, real.numExpertsPerToken);
        assertTrue(real.denseAttentionOnly());
        assertEquals("model.layers.19.shared_mlp.input_linear.weight", sharedMlpWeightName(19, "input_linear.weight"));
        assertEquals("model.layers.19.shared_mlp.output_linear.weight", sharedMlpWeightName(19, "output_linear.weight"));
    }

    @Test
    public void tinyCheckpointWritesScaledGraniteMoeHybridTensorShapes() throws Exception {
        GraniteMoeHybridConfig config = tinyConfig();
        Path modelDir = writeTinyCheckpoint(tempDir.resolve("granite-moe-hybrid-shapes"), config, 1334);
        try (DefaultWeightLoader loader = new DefaultWeightLoader(modelDir.toFile())) {
            Map<String, TensorInfo> info = loader.tensorInfoMap();
            assertShape(info, "model.embed_tokens.weight", config.vocabularySize, config.embeddingLength);
            assertShape(info, "lm_head.weight", config.vocabularySize, config.embeddingLength);
            assertShape(info, "model.layers.0.input_layernorm.weight", 1, config.embeddingLength);
            assertShape(info, "model.layers.0.self_attn.q_proj.weight", config.attentionLength, config.embeddingLength);
            assertShape(info, "model.layers.0.self_attn.k_proj.weight", config.kvLength, config.embeddingLength);
            assertShape(info, "model.layers.0.self_attn.v_proj.weight", config.kvLength, config.embeddingLength);
            assertShape(info, "model.layers.0.self_attn.o_proj.weight", config.embeddingLength, config.attentionLength);
            assertShape(info, "model.layers.0.post_attention_layernorm.weight", 1, config.embeddingLength);
            assertShape(info, sharedMlpWeightName(0, "input_linear.weight"), config.sharedIntermediateSize * 2,
                    config.embeddingLength);
            assertShape(info, sharedMlpWeightName(0, "output_linear.weight"), config.embeddingLength,
                    config.sharedIntermediateSize);
            assertShape(info, "model.norm.weight", 1, config.embeddingLength);
            assertTrue(info.keySet().stream().noneMatch(name -> name.contains("block_sparse_moe")));
        }
    }

    @Test
    public void tinyDenseAttentionOnlyModelLoads() {
        Path modelDir = tempDir.resolve("granite-moe-hybrid-tiny");
        writeTinyCheckpoint(modelDir, tinyConfig(), 2234);

        try (AbstractModel model = loadTinyModelStatic(modelDir)) {
            assertTrue(model instanceof GraniteMoeHybridModel);
            assertTrue(((GraniteMoeHybridConfig) model.getConfig()).denseAttentionOnly());
        }
    }

    @Test
    public void tinyMambaLayerConfigFailsClearly() {
        Path modelDir = tempDir.resolve("granite-moe-hybrid-mamba");
        writeTinyCheckpoint(modelDir, tinyConfigWithMambaLayer(), 3234);

        UnsupportedOperationException error = assertThrows(UnsupportedOperationException.class,
                () -> loadTinyModelStatic(modelDir));

        assertTrue(error.getMessage().contains("dense attention-only"));
    }

    @Test
    public void tinyMoeConfigFailsClearly() {
        Path modelDir = tempDir.resolve("granite-moe-hybrid-moe");
        writeTinyCheckpoint(modelDir, tinyConfigWithExperts(), 4234);

        UnsupportedOperationException error = assertThrows(UnsupportedOperationException.class,
                () -> loadTinyModelStatic(modelDir));

        assertTrue(error.getMessage().contains("dense attention-only"));
    }

    @Test
    public void tinyModelForwardReturnsExpectedShape() {
        Path modelDir = writeTinyCheckpoint(tempDir.resolve("granite-moe-hybrid-forward-shape"), tinyConfig(), 5234);
        try (GraniteMoeHybridModel model = loadTinyModelStatic(modelDir);
             KvBufferCache.KvBuffer kv = model.newKvBuffer();
             AbstractTensor output = model.batchForward(new int[]{3, 4, 5, 6}, 0, kv)) {
            assertEquals(4, output.shape().first());
            assertEquals(model.getConfig().embeddingLength, output.shape().last());
            assertFinite(output);
        }
    }

    @Test
    public void tinyModelForwardIsDeterministicForSameInput() {
        Path modelDir = writeTinyCheckpoint(tempDir.resolve("granite-moe-hybrid-deterministic"), tinyConfig(), 6234);
        int[] tokens = new int[]{3, 4, 5, 6};
        try (GraniteMoeHybridModel model = loadTinyModelStatic(modelDir);
             KvBufferCache.KvBuffer firstKv = model.newKvBuffer();
             KvBufferCache.KvBuffer secondKv = model.newKvBuffer();
             AbstractTensor first = model.batchForward(tokens, 0, firstKv);
             AbstractTensor second = model.batchForward(tokens, 0, secondKv)) {
            assertEquals(0.0f, drift(first, second).maxAbs(), "same model/input should be deterministic");
        }
    }

    @Test
    public void tinyModelDecodeMatchesColdReplayForFixedContinuation() {
        Path modelDir = writeTinyCheckpoint(tempDir.resolve("granite-moe-hybrid-decode"), tinyConfig(), 7234);
        int[] prompt = new int[]{3, 4, 5, 6};
        int[] continuation = new int[]{7, 8};
        try (GraniteMoeHybridModel model = loadTinyModelStatic(modelDir);
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

    static GraniteMoeHybridModel loadTinyModelStatic(Path modelDir) {
        MetricRegistry metrics = new MetricRegistry();
        TensorAllocator allocator = new ArrayQueueTensorAllocator(metrics);
        WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores());
        return new GraniteMoeHybridModel(AbstractModel.InferenceType.FULL_GENERATION, configFromFile(modelDir),
                new DefaultWeightLoader(modelDir.toFile()), Mockito.mock(PreTrainedTokenizer.class), DType.F32, DType.I8,
                Optional.empty(), new ConfigurableTensorProvider(new NaiveTensorOperations()), metrics, allocator,
                new KvBufferCacheSettings(true), new DefaultToolCallParser(), pool,
                new StaticTensorParallelContext(0, 1), new SingleRankTensorParallelCollectives(), Optional.empty());
    }

    static GraniteMoeHybridConfig tinyConfig() {
        return new GraniteMoeHybridConfig(64, 16, 32, 2, 1, 2, 1.0e-5f, 48, 1, 1,
                ActivationFunction.Type.SILU, 10_000.0, null, null, 0.0f, 2.0f, 0.25f, 0.5f,
                4.0f, 24, 0, 0, false, 0.01f, List.of("attention", "attention"), "rope",
                4, 1, 8, 8, 4, 2, 16, true, false, List.of("GraniteMoeHybridForCausalLM"));
    }

    static GraniteMoeHybridConfig tinyConfigWithMambaLayer() {
        return new GraniteMoeHybridConfig(64, 16, 32, 2, 1, 2, 1.0e-5f, 48, 1, 1,
                ActivationFunction.Type.SILU, 10_000.0, null, null, 0.0f, 2.0f, 0.25f, 0.5f,
                4.0f, 24, 0, 0, false, 0.01f, List.of("attention", "mamba"), "rope",
                4, 1, 8, 8, 4, 2, 16, true, false, List.of("GraniteMoeHybridForCausalLM"));
    }

    static GraniteMoeHybridConfig tinyConfigWithExperts() {
        return new GraniteMoeHybridConfig(64, 16, 32, 2, 1, 2, 1.0e-5f, 48, 1, 1,
                ActivationFunction.Type.SILU, 10_000.0, null, null, 0.0f, 2.0f, 0.25f, 0.5f,
                4.0f, 24, 2, 1, false, 0.01f, List.of("attention", "attention"), "rope",
                4, 1, 8, 8, 4, 2, 16, true, false, List.of("GraniteMoeHybridForCausalLM"));
    }

    static GraniteMoeHybridConfig realAntaresConfig() {
        return new GraniteMoeHybridConfig(131072, 2048, 4096, 16, 4, 40, 1.0e-5f, 100352, 100257,
                100257, ActivationFunction.Type.SILU, 10_000_000.0, null, null, 0.0f, 12.0f, 0.0078125f,
                0.22f, 8.0f, 4096, 0, 0, false, 0.01f, attentionLayers(40), "rope",
                128, 1, 256, 32, 4, 2, 256, true, false, List.of("GraniteMoeHybridForCausalLM"));
    }

    static Path writeTinyCheckpoint(Path dir, GraniteMoeHybridConfig config, int seed) {
        try {
            java.nio.file.Files.createDirectories(dir);
            JsonUtils.om.writeValue(dir.resolve("config.json").toFile(), tinyConfigJson(config));
            Map<String, AbstractTensor> tensors = new LinkedHashMap<>();
            tensors.put("model.embed_tokens.weight", matrix(config.vocabularySize, config.embeddingLength, seed++));
            tensors.put("lm_head.weight", matrix(config.vocabularySize, config.embeddingLength, seed++));
            tensors.put("model.norm.weight", ones(1, config.embeddingLength));
            for (int i = 0; i < config.numberOfLayers; i++) {
                String layer = "model.layers." + i + ".";
                tensors.put(layer + "input_layernorm.weight", ones(1, config.embeddingLength));
                tensors.put(layer + "post_attention_layernorm.weight", ones(1, config.embeddingLength));
                tensors.put(layer + "self_attn.q_proj.weight", matrix(config.attentionLength, config.embeddingLength, seed++));
                tensors.put(layer + "self_attn.k_proj.weight", matrix(config.kvLength, config.embeddingLength, seed++));
                tensors.put(layer + "self_attn.v_proj.weight", matrix(config.kvLength, config.embeddingLength, seed++));
                tensors.put(layer + "self_attn.o_proj.weight", matrix(config.embeddingLength, config.attentionLength, seed++));
                tensors.put(sharedMlpWeightName(i, "input_linear.weight"),
                        matrix(config.sharedIntermediateSize * 2, config.embeddingLength, seed++));
                tensors.put(sharedMlpWeightName(i, "output_linear.weight"),
                        matrix(config.embeddingLength, config.sharedIntermediateSize, seed++));
            }
            SafeTensorWriter.writeModel(dir, Map.of("format", "pt"), tensors, 1 << 28);
            java.nio.file.Files.createFile(dir.resolve(".finished"));
            tensors.values().forEach(AbstractTensor::close);
            return dir;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    static GraniteMoeHybridConfig configFromFile(Path modelDir) {
        try {
            return JsonUtils.om.readValue(modelDir.resolve("config.json").toFile(), GraniteMoeHybridConfig.class);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    static Map<String, Object> tinyConfigJson(GraniteMoeHybridConfig config) {
        Map<String, Object> json = new LinkedHashMap<>();
        json.put("model_type", "granitemoehybrid");
        json.put("architectures", List.of("GraniteMoeHybridForCausalLM"));
        json.put("max_position_embeddings", config.contextLength);
        json.put("hidden_size", config.embeddingLength);
        json.put("intermediate_size", config.hiddenLength);
        json.put("num_attention_heads", config.numberOfHeads);
        json.put("num_key_value_heads", config.numberOfKeyValueHeads);
        json.put("num_hidden_layers", config.numberOfLayers);
        json.put("rms_norm_eps", config.layerNormEps);
        json.put("vocab_size", config.vocabularySize);
        json.put("bos_token_id", config.bosToken);
        json.put("eos_token_id", config.eosTokens.getFirst());
        json.put("hidden_act", "silu");
        json.put("rope_theta", 10_000.0);
        json.put("position_embedding_type", config.positionEmbeddingType);
        json.put("attention_dropout", config.attentionDropout);
        json.put("embedding_multiplier", config.embeddingMultiplier);
        json.put("attention_multiplier", config.attentionMultiplier);
        json.put("residual_multiplier", config.residualMultiplier);
        json.put("logits_scaling", config.logitsScaling);
        json.put("shared_intermediate_size", config.sharedIntermediateSize);
        json.put("num_local_experts", config.numLocalExperts);
        json.put("num_experts_per_tok", config.numExpertsPerToken);
        json.put("output_router_logits", config.outputRouterLogits);
        json.put("router_aux_loss_coef", config.routerAuxLossCoef);
        json.put("layer_types", config.layerTypes);
        json.put("mamba_n_heads", config.mambaNHeads);
        json.put("mamba_n_groups", config.mambaNGroups);
        json.put("mamba_d_state", config.mambaDState);
        json.put("mamba_d_head", config.mambaDHead);
        json.put("mamba_d_conv", config.mambaDConv);
        json.put("mamba_expand", config.mambaExpand);
        json.put("mamba_chunk_size", config.mambaChunkSize);
        json.put("mamba_conv_bias", config.mambaConvBias);
        json.put("mamba_proj_bias", config.mambaProjBias);
        return json;
    }

    static String sharedMlpWeightName(int layer, String suffix) {
        return "model.layers." + layer + ".shared_mlp." + suffix;
    }

    private static List<String> attentionLayers(int layers) {
        return java.util.Collections.nCopies(layers, "attention");
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

    private static void assertShape(Map<String, TensorInfo> info, String name, int... expected) {
        TensorInfo tensorInfo = info.get(name);
        assertTrue(tensorInfo != null, "missing tensor " + name);
        assertEquals(java.util.Arrays.toString(expected), java.util.Arrays.toString(tensorInfo.shape), name);
    }

    private static void assertFinite(AbstractTensor tensor) {
        for (int row = 0; row < tensor.shape().first(); row++) {
            for (int col = 0; col < tensor.shape().last(); col++) {
                assertTrue(Float.isFinite(tensor.get(row, col)));
            }
        }
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

    private static AbstractTensor coldReplay(GraniteMoeHybridModel model, int[] prompt, int[] continuation,
            int continuationLength) {
        int[] tokens = java.util.Arrays.copyOf(prompt, prompt.length + continuationLength);
        System.arraycopy(continuation, 0, tokens, prompt.length, continuationLength);
        return model.batchForward(tokens, 0);
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

    private record Drift(float maxAbs, double meanAbs) {
    }
}
