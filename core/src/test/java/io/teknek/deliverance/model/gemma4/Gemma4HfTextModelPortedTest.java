package io.teknek.deliverance.model.gemma4;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.JsonUtils;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.grace.PreTrainedTokenizer;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.hf.HfConfigTesterMixinPort;
import io.teknek.deliverance.model.hf.HfGenerationTesterMixinPort;
import io.teknek.deliverance.model.hf.HfModelTesterMixinPort;
import io.teknek.deliverance.model.hf.HfUnsupportedMixinPort;
import io.teknek.deliverance.math.WrappedForkJoinPool;
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

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Ports the first Gemma4TextModelTest cases from Hugging Face.
 *
 * <p>Some upstream tests are intentionally skipped in HF. We keep those as disabled tests with the upstream reason so
 * the mismatch is explicit rather than forgotten.</p>
 */
public class Gemma4HfTextModelPortedTest implements
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
        return writeTinyCheckpoint(tempDir.resolve(name), fourLayerSharedKvConfig(), seed);
    }

    @Override
    public Gemma4Model loadTinyModel(Path modelDir) {
        return loadTinyGemma4Model(modelDir);
    }

    @Override
    public Gemma4Config loadTinyConfig(Path modelDir) {
        return configFromFile(modelDir);
    }

    @Override
    public Config roundTripConfig(Config config) throws Exception {
        Gemma4Config gemma4Config = (Gemma4Config) config;
        String json = JsonUtils.om.writeValueAsString(Map.of(
                "model_type", "gemma4",
                "architectures", gemma4Config.architectures,
                "eos_token_id", gemma4Config.eosTokens,
                "text_config", tinyTextConfig(gemma4Config)
        ));
        return JsonUtils.om.readValue(json, Gemma4Config.class);
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
        Gemma4Config first = (Gemma4Config) expected;
        Gemma4Config second = (Gemma4Config) actual;
        assertEquals(first.layerTypes, second.layerTypes);
        assertEquals(first.ropeParametersByLayerType, second.ropeParametersByLayerType);
        assertEquals(first.numKvSharedLayers, second.numKvSharedLayers);
        assertEquals(first.hiddenSizePerLayerInput, second.hiddenSizePerLayerInput);
        assertEquals(first.vocabSizePerLayerInput, second.vocabSizePerLayerInput);
    }

    @Test
    public void testNumLayersIsSmall() {
        Gemma4Config config = fourLayerSharedKvConfig();

        assertEquals(4, config.numberOfLayers);
        assertEquals(2, config.numKvSharedLayers);
        assertEquals(List.of("sliding_attention", "full_attention", "sliding_attention", "full_attention"),
                config.layerTypes);
        assertEquals(0, config.getSharedKvSourceLayer(2));
        assertEquals(1, config.getSharedKvSourceLayer(3));
        assertTrue(config.storesSharedKvState(0));
        assertTrue(config.storesSharedKvState(1));
    }

    @Test
    public void fourLayerSetupStoresAndConsumesSharedKvStates() {
        Path modelDir = writeTinyCheckpoint(tempDir.resolve("gemma4-tiny"), fourLayerSharedKvConfig(), 1234);
        try (Gemma4Model model = loadTinyModel(modelDir);
             KvBufferCache.KvBuffer kv = model.newKvBuffer();
             AbstractTensor output = model.batchForward(new int[]{3, 4, 5}, 0, kv)) {
            assertEquals(3, output.shape().first());
            assertEquals(model.getConfig().embeddingLength, output.shape().last());
        }
    }

    @Test
    public void hfTextModelTesterMoeExpertsAffectForwardOutput() {
        Gemma4Config config = fourLayerSharedKvMoeConfig();
        Path firstModelDir = writeTinyCheckpoint(tempDir.resolve("gemma4-tiny-moe-first"), config, 12_001);
        Path secondModelDir = writeTinyCheckpoint(tempDir.resolve("gemma4-tiny-moe-second"), config, 12_001, 99_001);
        int[] tokens = new int[]{3, 4, 5, 6};

        try (Gemma4Model firstModel = loadTinyModel(firstModelDir);
             Gemma4Model secondModel = loadTinyModel(secondModelDir);
             KvBufferCache.KvBuffer firstKv = firstModel.newKvBuffer();
             KvBufferCache.KvBuffer secondKv = secondModel.newKvBuffer();
             AbstractTensor first = firstModel.batchForward(tokens, 0, firstKv);
             AbstractTensor second = secondModel.batchForward(tokens, 0, secondKv)) {
            Drift drift = drift(first, second);
            assertTrue(drift.maxAbs() > 1.0e-6f,
                    "with enable_moe_block=true, different router/expert weights should affect output: " + drift);
        }
    }

    @Test
    public void denseGemma4ConfigDoesNotRequireMoeWeights() {
        Gemma4Config config = fourLayerSharedKvDenseConfig();
        Path modelDir = writeTinyCheckpoint(tempDir.resolve("gemma4-tiny-dense-no-moe"), config, 12_101);
        try (Gemma4Model model = loadTinyModel(modelDir);
             KvBufferCache.KvBuffer kv = model.newKvBuffer();
             AbstractTensor output = model.batchForward(new int[]{3, 4, 5, 6}, 0, kv)) {
            assertEquals(4, output.shape().first());
            assertEquals(model.getConfig().embeddingLength, output.shape().last());
            assertFinite(output);
        }
    }

    @Test
    public void tinyModelForwardReturnsExpectedShape() {
        Path modelDir = writeTinyCheckpoint(tempDir.resolve("gemma4-tiny-shape"), fourLayerSharedKvConfig(), 2234);
        try (Gemma4Model model = loadTinyModel(modelDir);
             KvBufferCache.KvBuffer kv = model.newKvBuffer();
             AbstractTensor output = model.batchForward(new int[]{3, 4, 5, 6}, 0, kv)) {
            assertEquals(4, output.shape().first());
            assertEquals(model.getConfig().embeddingLength, output.shape().last());
        }
    }

    @Test
    public void tinyModelForwardIsDeterministicForSameInput() {
        Path modelDir = writeTinyCheckpoint(tempDir.resolve("gemma4-tiny-deterministic"), fourLayerSharedKvConfig(), 3234);
        int[] tokens = new int[]{3, 4, 5, 6};
        try (Gemma4Model model = loadTinyModel(modelDir);
             KvBufferCache.KvBuffer firstKv = model.newKvBuffer();
             KvBufferCache.KvBuffer secondKv = model.newKvBuffer();
             AbstractTensor first = model.batchForward(tokens, 0, firstKv);
             AbstractTensor second = model.batchForward(tokens, 0, secondKv)) {
            Drift drift = drift(first, second);
            assertEquals(0.0f, drift.maxAbs(), "same model/input should be deterministic: " + drift);
        }
    }

    @Test
    public void tinyModelBatchPrefillMatchesTokenByTokenPrefill() {
        Path modelDir = writeTinyCheckpoint(tempDir.resolve("gemma4-tiny-prefill"), fourLayerSharedKvConfig(), 4234);
        int[] tokens = new int[]{3, 4, 5, 6};
        try (Gemma4Model model = loadTinyModel(modelDir);
             KvBufferCache.KvBuffer batchKv = model.newKvBuffer();
             KvBufferCache.KvBuffer stepKv = model.newKvBuffer();
             AbstractTensor batchOutput = model.batchForward(tokens, 0, batchKv);
             AbstractTensor stepOutput = tokenByTokenPrefill(model, tokens, stepKv)) {
            Drift drift = driftLastBatchRow(batchOutput, stepOutput);
            assertTrue(drift.maxAbs() < 1.0e-4f, "batch prefill should match token prefill: " + drift);
        }
    }

    @Test
    public void tinyModelDecodeMatchesColdReplayForFixedContinuation() {
        Path modelDir = writeTinyCheckpoint(tempDir.resolve("gemma4-tiny-decode"), fourLayerSharedKvConfig(), 5234);
        int[] prompt = new int[]{3, 4, 5, 6};
        int[] continuation = new int[]{7, 8, 9};
        try (Gemma4Model model = loadTinyModel(modelDir);
             KvBufferCache.KvBuffer decodeKv = model.newKvBuffer();
             AbstractTensor promptOutput = model.batchForward(prompt, 0, decodeKv)) {
            promptOutput.close();
            for (int i = 0; i < continuation.length; i++) {
                try (AbstractTensor decode = model.forward(continuation[i], prompt.length + i, decodeKv);
                     AbstractTensor replay = coldReplay(model, prompt, continuation, i + 1)) {
                    Drift drift = driftLastBatchRow(replay, decode);
                    assertTrue(drift.maxAbs() < 1.0e-4f, "decode should match cold replay at step " + i + ": " + drift);
                }
            }
        }
    }

    @Test
    public void testModelTextOnly() {
        Path modelDir = writeTinyCheckpoint(tempDir.resolve("gemma4-tiny-text-only"), fourLayerSharedKvConfig(), 8234);
        try (Gemma4Model model = loadTinyModel(modelDir);
             KvBufferCache.KvBuffer kv = model.newKvBuffer();
             AbstractTensor output = model.batchForward(new int[]{3, 4, 5, 6, 7}, 0, kv)) {
            assertEquals(5, output.shape().first());
            assertEquals(model.getConfig().embeddingLength, output.shape().last());
            assertFinite(output);
        }
    }

    @Test
    public void testStatesSharingWithAndWithoutCache() {
        Path modelDir = writeTinyCheckpoint(tempDir.resolve("gemma4-tiny-cache-sharing"), fourLayerSharedKvConfig(), 9234);
        int[] prompt = new int[]{3, 4, 5, 6, 7, 8};
        int[] continuation = new int[]{9, 10, 11, 12};
        try (Gemma4Model model = loadTinyModel(modelDir)) {
            assertDecodeMatchesColdReplay(model, prompt, continuation, 1.0e-4f);
        }
    }

    @Test
    public void testGenerationBeyondSlidingWindow() {
        Path modelDir = writeTinyCheckpoint(tempDir.resolve("gemma4-tiny-beyond-window"), fourLayerSharedKvConfig(), 10_234);
        int[] prompt = new int[24];
        for (int i = 0; i < prompt.length; i++) {
            prompt[i] = 3 + (i % 20);
        }
        int[] continuation = new int[]{31, 32, 33};
        try (Gemma4Model model = loadTinyModel(modelDir);
             KvBufferCache.KvBuffer kv = model.newKvBuffer();
             AbstractTensor output = model.batchForward(prompt, 0, kv)) {
            assertEquals(prompt.length, output.shape().first());
            assertFinite(output);
            output.close();
            assertDecodeMatchesColdReplay(model, prompt, continuation, 1.0e-4f);
        }
    }

    @Test
    public void tinyModelRejectsInputsEmbedsForwardWhenPleIsEnabled() {
        Path modelDir = writeTinyCheckpoint(tempDir.resolve("gemma4-tiny-inputs-embeds"), fourLayerSharedKvConfig(), 6234);
        try (Gemma4Model model = loadTinyModel(modelDir);
             KvBufferCache.KvBuffer kv = model.newKvBuffer();
             AbstractTensor inputsEmbeds = matrix(1, model.getConfig().embeddingLength, 1)) {
            assertThrows(UnsupportedOperationException.class, () -> model.forward(inputsEmbeds, 0, kv, Optional.empty()));
        }
    }

    @Test
    @Disabled("HF skips this: Gemma4 uses different RoPE per layer type, which is not compatible with this common test")
    public void testModelRopeScalingFrequencies() {
    }

    @Test
    @Disabled("HF skips this: Gemma4 uses different RoPE per layer type, which is not compatible with this common test")
    public void testModelRopeScalingFromConfig() {
    }

    @Test
    @Disabled("HF skips this: Gemma4 cannot use random inputs_embeds, as it needs to reverse them when input_ids is not provided")
    public void testGenerateFromRandomInputsEmbeds() {
        Path modelDir = writeTinyCheckpoint(tempDir.resolve("gemma4-tiny-random-inputs-embeds"), fourLayerSharedKvConfig(), 6235);
        int[] inputIds = new int[]{3, 4, 5, 6};
        try (Gemma4Model model = loadTinyModel(modelDir);
             KvBufferCache.KvBuffer tokenKv = model.newKvBuffer();
             KvBufferCache.KvBuffer embedsKv = model.newKvBuffer();
             KvBufferCache.KvBuffer randomEmbedsKv = model.newKvBuffer();
             AbstractTensor tokenOutput = model.batchForward(inputIds, 0, tokenKv);
             AbstractTensor inputsEmbeds = matrix(inputIds.length, model.getConfig().embeddingLength, 6236);
             AbstractTensor randomEmbeds = matrix(inputIds.length, model.getConfig().embeddingLength, 6237);
             AbstractTensor outputsFromEmbeds = model.forward(inputsEmbeds, 0, embedsKv, Optional.empty());
             AbstractTensor outputsFromRandomEmbeds = model.forward(randomEmbeds, 0, randomEmbedsKv, Optional.empty())) {
            assertFinite(tokenOutput);
            assertFinite(outputsFromEmbeds);
            assertFinite(outputsFromRandomEmbeds);

            Drift drift = drift(outputsFromEmbeds, outputsFromRandomEmbeds);
            assertTrue(drift.maxAbs() > 1.0e-6f,
                    "different inputs_embeds should produce different outputs, matching HF GenerationTesterMixin intent: " + drift);
        }
    }

    @Test
    @Disabled("HF skips this: flaky under bf16/fp32 precision differences; TODO upstream investigates precision source")
    public void testSdpaPaddingMatchesPaddingFreeWithPositionIds() {
    }

    @Test
    @Disabled("HF skips this: fails after fully removing unused weights; upstream TODO investigates why")
    public void testTpGenerationQuantized() {
    }

    @Test
    @Disabled("HF skips this: randomly initialized Gemma4 MoE routers are too sensitive to tiny eager/FA2 input differences")
    public void testFlashAttn2Equivalence() {
    }

    @Test
    @Disabled("HF skips this: randomly initialized Gemma4 MoE routers are too sensitive to tiny eager/FA2 input differences")
    public void testFlashAttn2InferenceEquivalence() {
    }

    @Test
    @Disabled("HF skips this: randomly initialized Gemma4 MoE routers are too sensitive to tiny eager/FA2 input differences")
    public void testFlashAttn2InferenceEquivalenceRightPadding() {
    }

    @Test
    public void testAllBidirectionalAttentionUsesBidirectionalMask() {
        Gemma4Config config = fourLayerSharedKvConfig("all");
        Path modelDir = writeTinyCheckpoint(tempDir.resolve("gemma4-tiny-bidir"), config, 7234);
        try (Gemma4Model model = loadTinyModel(modelDir);
             KvBufferCache.KvBuffer firstKv = model.newKvBuffer();
             KvBufferCache.KvBuffer secondKv = model.newKvBuffer();
             AbstractTensor first = model.batchForward(new int[]{3, 4, 5, 6}, 0, firstKv);
             AbstractTensor second = model.batchForward(new int[]{3, 4, 5, 7}, 0, secondKv)) {
            Drift drift = driftFirstBatchRow(first, second);
            assertTrue(drift.maxAbs() > 1.0e-6f,
                    "with all-bidirectional attention, changing a future token should affect the first token: " + drift);
        }
    }

    @Test
    public void testVisionBidirectionalAttentionKeepsTextCausalMask() {
        Gemma4Config config = fourLayerSharedKvConfig("vision");
        Path modelDir = writeTinyCheckpoint(tempDir.resolve("gemma4-tiny-vision-causal"), config, 7235);
        try (Gemma4Model model = loadTinyModel(modelDir);
             KvBufferCache.KvBuffer firstKv = model.newKvBuffer();
             KvBufferCache.KvBuffer secondKv = model.newKvBuffer();
             AbstractTensor first = model.batchForward(new int[]{3, 4, 5, 6}, 0, firstKv);
             AbstractTensor second = model.batchForward(new int[]{3, 4, 5, 7}, 0, secondKv)) {
            Drift drift = driftFirstBatchRow(first, second);
            assertEquals(0.0f, drift.maxAbs(), 1.0e-6f,
                    "vision-bidirectional mode should not let a future text token affect the first text token: " + drift);
        }
    }

    @Test
    public void testModelTraining() {
        // HF overrides this as a no-op for Gemma4TextModelTest.
    }

    @Test
    @Disabled("HF skips this: non-bf16 MoE grouped_mm fallback is incompatible with torch.compile reduce-overhead")
    public void testFlashAttn2CanCompileWithAttentionMaskNoneWithoutGraphBreak() {
    }

    @Test
    @Disabled("HF skips this: non-bf16 MoE grouped_mm fallback is incompatible with torch.compile reduce-overhead")
    public void testTorchCompileForTraining() {
    }

    @Test
    @Disabled("HF text-only multigpu test requires Accelerate/device_map; Deliverance has no equivalent local unit test")
    public void testModelTextOnlyMultigpu() {
    }

    @Test
    @Disabled("HF export test targets TorchExportableModuleForDecoderOnlyLM; Deliverance has no torch export equivalent")
    public void testExportTextOnly() {
    }

    @Test
    @Disabled("HF verifies explicit per_layer_inputs API forwarding; Deliverance Gemma4 currently computes PLE internally and has no public per_layer_inputs forward API")
    public void testPerLayerInputsAreCorrectlyForwarded() {
    }

    static Gemma4Config fourLayerSharedKvConfig() {
        return fourLayerSharedKvConfig(null);
    }

    static Gemma4Config fourLayerSharedKvMoeConfig() {
        Map<String, Object> textConfig = tinyTextConfig((String) null);
        textConfig.put("enable_moe_block", true);
        textConfig.put("num_experts", 8);
        textConfig.put("top_k_experts", 2);
        textConfig.put("moe_intermediate_size", 16);
        return new Gemma4Config(textConfig, List.of("Gemma4ForConditionalGeneration"), List.of(2));
    }

    static Gemma4Config fourLayerSharedKvDenseConfig() {
        Map<String, Object> textConfig = tinyTextConfig((String) null);
        textConfig.put("enable_moe_block", false);
        textConfig.remove("num_experts");
        textConfig.remove("top_k_experts");
        textConfig.remove("moe_intermediate_size");
        return new Gemma4Config(textConfig, List.of("Gemma4ForConditionalGeneration"), List.of(2));
    }

    static Gemma4Config fourLayerSharedKvConfig(String useBidirectionalAttention) {
        return new Gemma4Config(tinyTextConfig(useBidirectionalAttention),
                List.of("Gemma4ForConditionalGeneration"), List.of(2));
    }

    static Gemma4Model loadTinyGemma4Model(Path modelDir) {
        MetricRegistry metrics = new MetricRegistry();
        TensorAllocator allocator = new ArrayQueueTensorAllocator(metrics);
        WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores());
        ConfigurableTensorProvider provider = new ConfigurableTensorProvider(new NaiveTensorOperations());
        return new Gemma4Model(
                AbstractModel.InferenceType.FULL_GENERATION,
                configFromFile(modelDir),
                new DefaultWeightLoader(modelDir.toFile()),
                Mockito.mock(PreTrainedTokenizer.class),
                DType.F32,
                DType.I8,
                Optional.empty(),
                provider,
                metrics,
                allocator,
                new KvBufferCacheSettings(true),
                new DefaultToolCallParser(),
                pool,
                new StaticTensorParallelContext(0, 1),
                new SingleRankTensorParallelCollectives(),
                Optional.empty()
        );
    }

    static Path writeTinyCheckpoint(Path dir, Gemma4Config config, int seed) {
        return writeTinyCheckpoint(dir, config, seed, seed);
    }

    static Path writeTinyCheckpoint(Path dir, Gemma4Config config, int seed, int moeSeed) {
        try {
            java.nio.file.Files.createDirectories(dir);
            JsonUtils.om.writeValue(dir.resolve("config.json").toFile(), Map.of(
                    "model_type", "gemma4",
                    "architectures", List.of("Gemma4ForConditionalGeneration"),
                    "eos_token_id", config.eosTokens,
                    "text_config", tinyTextConfig(config)
            ));
            Map<String, AbstractTensor> tensors = new LinkedHashMap<>();
            String root = "model.language_model.";
            int perLayerVocab = config.vocabSizePerLayerInput == null ? config.vocabularySize : config.vocabSizePerLayerInput;
            tensors.put(root + "embed_tokens.weight", matrix(config.vocabularySize, config.embeddingLength, seed++));
            tensors.put(root + "norm.weight", ones(1, config.embeddingLength));
            tensors.put(root + "embed_tokens_per_layer.weight", matrix(perLayerVocab,
                    config.numberOfLayers * config.hiddenSizePerLayerInput, seed++));
            tensors.put(root + "per_layer_model_projection.weight", matrix(config.numberOfLayers * config.hiddenSizePerLayerInput,
                    config.embeddingLength, seed++));
            tensors.put(root + "per_layer_projection_norm.weight", ones(1, config.hiddenSizePerLayerInput));
            for (int i = 0; i < config.numberOfLayers; i++) {
                String layer = root + "layers." + i + ".";
                String layerType = config.layerTypes.get(i);
                boolean shared = config.getSharedKvSourceLayer(i) >= 0;
                int queryLength = config.getLayerQueryProjectionLength(layerType);
                int kvLength = config.getLayerKeyValueProjectionLength(layerType);
                int hiddenLength = config.getLayerHiddenLength(i);
                tensors.put(layer + "input_layernorm.weight", ones(1, config.embeddingLength));
                tensors.put(layer + "post_attention_layernorm.weight", ones(1, config.embeddingLength));
                tensors.put(layer + "pre_feedforward_layernorm.weight", ones(1, config.embeddingLength));
                tensors.put(layer + "post_feedforward_layernorm.weight", ones(1, config.embeddingLength));
                tensors.put(layer + "self_attn.q_proj.weight", matrix(queryLength, config.embeddingLength, seed++));
                tensors.put(layer + "self_attn.q_norm.weight", ones(1, config.getLayerHeadDim(layerType)));
                if (!shared) {
                    tensors.put(layer + "self_attn.k_proj.weight", matrix(kvLength, config.embeddingLength, seed++));
                    tensors.put(layer + "self_attn.v_proj.weight", matrix(kvLength, config.embeddingLength, seed++));
                    tensors.put(layer + "self_attn.k_norm.weight", ones(1, config.getLayerHeadDim(layerType)));
                }
                tensors.put(layer + "self_attn.o_proj.weight", matrix(config.embeddingLength, queryLength, seed++));
                tensors.put(layer + "mlp.gate_proj.weight", matrix(hiddenLength, config.embeddingLength, seed++));
                tensors.put(layer + "mlp.up_proj.weight", matrix(hiddenLength, config.embeddingLength, seed++));
                tensors.put(layer + "mlp.down_proj.weight", matrix(config.embeddingLength, hiddenLength, seed++));
                if (config.enableMoeBlock) {
                    tensors.put(layer + "router.proj.weight", matrix(config.numExperts, config.embeddingLength, moeSeed++));
                    tensors.put(layer + "router.scale", ones(1, config.embeddingLength));
                    tensors.put(layer + "router.per_expert_scale", ones(1, config.numExperts));
                    tensors.put(layer + "experts.gate_up_proj", tensor3d(config.numExperts,
                            2 * config.moeIntermediateSize, config.embeddingLength, moeSeed++));
                    tensors.put(layer + "experts.down_proj", tensor3d(config.numExperts,
                            config.embeddingLength, config.moeIntermediateSize, moeSeed++));
                    tensors.put(layer + "post_feedforward_layernorm_1.weight", ones(1, config.embeddingLength));
                    tensors.put(layer + "pre_feedforward_layernorm_2.weight", ones(1, config.embeddingLength));
                    tensors.put(layer + "post_feedforward_layernorm_2.weight", ones(1, config.embeddingLength));
                }
                tensors.put(layer + "per_layer_input_gate.weight", matrix(config.hiddenSizePerLayerInput, config.embeddingLength, seed++));
                tensors.put(layer + "per_layer_projection.weight", matrix(config.embeddingLength, config.hiddenSizePerLayerInput, seed++));
                tensors.put(layer + "post_per_layer_input_norm.weight", ones(1, config.embeddingLength));
            }
            SafeTensorWriter.writeModel(dir, Map.of("format", "pt"), tensors, 1 << 28);
            tensors.values().forEach(AbstractTensor::close);
            return dir;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static Map<String, Object> tinyTextConfig() {
        return tinyTextConfig((String) null);
    }

    private static Map<String, Object> tinyTextConfig(String useBidirectionalAttention) {
        Map<String, Object> textConfig = new LinkedHashMap<>(Map.ofEntries(
                Map.entry("max_position_embeddings", 32),
                Map.entry("hidden_size", 32),
                Map.entry("intermediate_size", 32),
                Map.entry("num_attention_heads", 2),
                Map.entry("num_key_value_heads", 2),
                Map.entry("num_hidden_layers", 4),
                Map.entry("num_kv_shared_layers", 2),
                Map.entry("rms_norm_eps", 1.0e-6),
                Map.entry("vocab_size", 99),
                Map.entry("bos_token_id", 1),
                Map.entry("eos_token_id", List.of(2)),
                Map.entry("hidden_activation", "gelu_pytorch_tanh"),
                Map.entry("head_dim", 16),
                Map.entry("global_head_dim", 16),
                Map.entry("hidden_size_per_layer_input", 16),
                Map.entry("vocab_size_per_layer_input", 99),
                Map.entry("sliding_window", 16),
                Map.entry("use_bidirectional_attention", "vision"),
                Map.entry("enable_moe_block", true),
                Map.entry("num_experts", 8),
                Map.entry("top_k_experts", 2),
                Map.entry("moe_intermediate_size", 16),
                Map.entry("layer_types", List.of("sliding_attention", "full_attention", "sliding_attention", "full_attention")),
                Map.entry("rope_parameters", Map.of(
                        "sliding_attention", Map.of("rope_theta", 10000.0),
                        "full_attention", Map.of("rope_theta", 1000000.0, "rope_type", "proportional", "partial_rotary_factor", 0.25)
                ))
        ));
        if (useBidirectionalAttention != null) {
            textConfig.put("use_bidirectional_attention", useBidirectionalAttention);
        }
        return textConfig;
    }

    private static Map<String, Object> tinyTextConfig(Gemma4Config config) {
        Map<String, Object> textConfig = tinyTextConfig(config.useBidirectionalAttention);
        textConfig.put("enable_moe_block", config.enableMoeBlock);
        textConfig.put("use_double_wide_mlp", config.useDoubleWideMlp);
        if (config.numExperts != null) {
            textConfig.put("num_experts", config.numExperts);
        }
        if (config.topKExperts != null) {
            textConfig.put("top_k_experts", config.topKExperts);
        }
        if (config.moeIntermediateSize != null) {
            textConfig.put("moe_intermediate_size", config.moeIntermediateSize);
        }
        return textConfig;
    }

    static Gemma4Config configFromFile(Path modelDir) {
        try {
            return JsonUtils.om.readValue(modelDir.resolve("config.json").toFile(), Gemma4Config.class);
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

    private static FloatBufferTensor tensor3d(int d0, int d1, int d2, int seed) {
        FloatBufferTensor tensor = new FloatBufferTensor(d0, d1, d2);
        for (int i = 0; i < d0; i++) {
            for (int j = 0; j < d1; j++) {
                for (int k = 0; k < d2; k++) {
                    tensor.set(((i * 11 + j * 13 + k * 7 + seed) % 17 - 8) / 8.0f, i, j, k);
                }
            }
        }
        return tensor;
    }

    private static FloatBufferTensor ones(int rows, int cols) {
        FloatBufferTensor tensor = new FloatBufferTensor(rows, cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                tensor.set(1.0f, row, col);
            }
        }
        return tensor;
    }

    private static AbstractTensor tokenByTokenPrefill(Gemma4Model model, int[] tokens, KvBufferCache.KvBuffer kv) {
        AbstractTensor output = null;
        for (int i = 0; i < tokens.length; i++) {
            if (output != null) {
                output.close();
            }
            output = model.forward(tokens[i], i, kv);
        }
        return output;
    }

    private static AbstractTensor coldReplay(Gemma4Model model, int[] prompt, int[] continuation, int continuationLength) {
        int[] tokens = java.util.Arrays.copyOf(prompt, prompt.length + continuationLength);
        System.arraycopy(continuation, 0, tokens, prompt.length, continuationLength);
        return model.batchForward(tokens, 0);
    }

    private static void assertDecodeMatchesColdReplay(Gemma4Model model, int[] prompt, int[] continuation, float tolerance) {
        try (KvBufferCache.KvBuffer decodeKv = model.newKvBuffer();
             AbstractTensor promptOutput = model.batchForward(prompt, 0, decodeKv)) {
            promptOutput.close();
            for (int i = 0; i < continuation.length; i++) {
                try (AbstractTensor decode = model.forward(continuation[i], prompt.length + i, decodeKv);
                     AbstractTensor replay = coldReplay(model, prompt, continuation, i + 1)) {
                    Drift drift = driftLastBatchRow(replay, decode);
                    assertTrue(drift.maxAbs() < tolerance, "decode should match cold replay at step " + i + ": " + drift);
                }
            }
        }
    }

    private static void assertFinite(AbstractTensor tensor) {
        for (int row = 0; row < tensor.shape().first(); row++) {
            for (int col = 0; col < tensor.shape().last(); col++) {
                float value = tensor.get(row, col);
                assertTrue(Float.isFinite(value), "non-finite value row=" + row + " col=" + col + " value=" + value);
            }
        }
    }

    private static Drift driftLastBatchRow(AbstractTensor batchOutput, AbstractTensor singleRowOutput) {
        assertEquals(1, singleRowOutput.shape().first());
        assertEquals(batchOutput.shape().last(), singleRowOutput.shape().last());
        int row = batchOutput.shape().first() - 1;
        double total = 0.0;
        float max = 0.0f;
        for (int col = 0; col < batchOutput.shape().last(); col++) {
            float diff = Math.abs(batchOutput.get(row, col) - singleRowOutput.get(0, col));
            max = Math.max(max, diff);
            total += diff;
        }
        return new Drift(max, total / batchOutput.shape().last());
    }

    private static Drift driftFirstBatchRow(AbstractTensor first, AbstractTensor second) {
        assertEquals(first.shape().first(), second.shape().first());
        assertEquals(first.shape().last(), second.shape().last());
        double total = 0.0;
        float max = 0.0f;
        for (int col = 0; col < first.shape().last(); col++) {
            float diff = Math.abs(first.get(0, col) - second.get(0, col));
            max = Math.max(max, diff);
            total += diff;
        }
        return new Drift(max, total / first.shape().last());
    }

    private static Drift drift(AbstractTensor first, AbstractTensor second) {
        assertEquals(first.shape().first(), second.shape().first());
        assertEquals(first.shape().last(), second.shape().last());
        double total = 0.0;
        float max = 0.0f;
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
