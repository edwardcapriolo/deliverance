package io.teknek.deliverance.model.llama;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.JsonUtils;
import io.teknek.deliverance.grace.PreTrainedTokenizer;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.GenerationEngineRef;
import io.teknek.deliverance.model.ResponseContext;
import io.teknek.deliverance.model.SamplerReturn;
import io.teknek.deliverance.model.hf.HfConfigTesterMixinPort;
import io.teknek.deliverance.model.hf.HfModelTesterMixinPort;
import io.teknek.deliverance.model.tensorparallel.SingleRankTensorParallelCollectives;
import io.teknek.deliverance.model.tensorparallel.StaticTensorParallelContext;
import io.teknek.deliverance.model.tensorparallel.TensorParallelCollectives;
import io.teknek.deliverance.model.tensorparallel.TensorParallelContext;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.safetensors.DefaultWeightLoader;
import io.teknek.deliverance.safetensors.LoraAdapter;
import io.teknek.deliverance.safetensors.LoraAdapterConfig;
import io.teknek.deliverance.safetensors.SafeTensorWriter;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.KvBufferCache;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.TensorInfo;
import io.teknek.deliverance.tensor.kv.KvCacheSession;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.impl.Q4ByteBufferTensor;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.MachineSpec;
import io.teknek.deliverance.tensor.operations.NaiveTensorOperations;
import io.teknek.deliverance.tensor.operations.PanamaTensorOperations;
import io.teknek.deliverance.tensor2.TensorProviderKind;
import io.teknek.deliverance.tensor2.TensorRef;
import io.teknek.deliverance.tensor2.NativeOps;
import io.teknek.deliverance.toolcallparser.DefaultToolCallParser;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.mockito.Mockito;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Arrays;
import java.util.concurrent.ForkJoinPool;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Ports the feasible Llama text-model checks from Hugging Face.
 *
 * <p>Source: {@code /ai-code/transformers/tests/models/llama/test_modeling_llama.py}.</p>
 * The upstream {@code LlamaModelTest} inherits its runnable coverage from
 * {@code CausalLMModelTest}, {@code ModelTesterMixin}, and the config tester. This class ports the
 * shape, forward, deterministic, and different-weight checks against a tiny real-layout checkpoint.
 * Accelerator, torch compile/export, gated-checkpoint, and attention-backend cases remain in the
 * existing disabled inventory classes.</p>
 */
public class LlamaHfTextModelPortedTest implements HfConfigTesterMixinPort, HfModelTesterMixinPort {
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
    public LlamaModel loadTinyModel(Path modelDir) {
        return loadLlamaModel(modelDir, true);
    }

    @Override
    public LlamaConfig loadTinyConfig(Path modelDir) {
        try {
            return JsonUtils.om.readValue(modelDir.resolve("config.json").toFile(), LlamaConfig.class);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public Config roundTripConfig(Config config) throws Exception {
        return JsonUtils.om.readValue(JsonUtils.om.writeValueAsString(tinyConfigJson((LlamaConfig) config)),
                LlamaConfig.class);
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
        LlamaConfig first = (LlamaConfig) expected;
        LlamaConfig second = (LlamaConfig) actual;
        assertEquals(first.activationFunction, second.activationFunction);
    }

    @Test
    public void tinyConfigHasLlamaShape() {
        LlamaConfig config = tinyConfig();
        assertEquals(32, config.contextLength);
        assertEquals(32, config.embeddingLength);
        assertEquals(64, config.hiddenLength);
        assertEquals(2, config.numberOfHeads);
        assertEquals(1, config.numberOfKeyValueHeads);
        assertEquals(2, config.numberOfLayers);
        assertEquals(16, config.headSize);
        assertEquals(32, config.attentionLength);
        assertEquals(16, config.kvLength);
    }

    @Test
    public void tinyCheckpointWritesRealLlamaTensorNamesAndShapes() throws Exception {
        LlamaConfig config = tinyConfig();
        Path modelDir = writeTinyCheckpoint(tempDir.resolve("llama-tiny-layout"), config, 100);
        try (DefaultWeightLoader loader = new DefaultWeightLoader(modelDir.toFile())) {
            Map<String, TensorInfo> info = loader.tensorInfoMap();
            assertShape(info, "model.embed_tokens.weight", config.vocabularySize, config.embeddingLength);
            assertShape(info, "model.layers.0.input_layernorm.weight", 1, config.embeddingLength);
            assertShape(info, "model.layers.0.self_attn.q_proj.weight", config.attentionLength, config.embeddingLength);
            assertShape(info, "model.layers.0.self_attn.k_proj.weight", config.kvLength, config.embeddingLength);
            assertShape(info, "model.layers.0.self_attn.v_proj.weight", config.kvLength, config.embeddingLength);
            assertShape(info, "model.layers.0.self_attn.o_proj.weight", config.embeddingLength, config.attentionLength);
            assertShape(info, "model.layers.0.post_attention_layernorm.weight", 1, config.embeddingLength);
            assertShape(info, "model.layers.0.mlp.gate_proj.weight", config.hiddenLength, config.embeddingLength);
            assertShape(info, "model.layers.0.mlp.up_proj.weight", config.hiddenLength, config.embeddingLength);
            assertShape(info, "model.layers.0.mlp.down_proj.weight", config.embeddingLength, config.hiddenLength);
            assertShape(info, "model.norm.weight", 1, config.embeddingLength);
            assertTrue(info.keySet().stream().noneMatch(name -> name.contains("query_key_value")));
        }
    }

    @Test
    public void tinyLegacyAndTensorRefPrefillOutputsMatch() {
        Path modelDir = writeTinyCheckpoint(tempDir.resolve("llama-tiny-prefill-parity"), tinyConfig(), 200);
        try (LlamaModel tensorRef = loadLlamaModel(modelDir, true);
             LlamaModel legacy = loadLlamaModel(modelDir, false);
             AbstractTensor refOutput = tensorRef.batchForward(hfSampleTokenIds(), 0);
             AbstractTensor legacyOutput = legacy.batchForward(hfSampleTokenIds(), 0)) {
            assertPrefillTensorsClose(legacyOutput, refOutput, 1.0e-3f, "legacy/TensorRef prefill");
        }
    }

    @Test
    public void tinyLegacyAndTensorRefStageTraceMatches() {
        Path modelDir = writeTinyCheckpoint(tempDir.resolve("llama-tiny-stage-parity"), tinyConfig(), 201);
        Map<String, float[]> tensorRefTrace = new LinkedHashMap<>();
        Map<String, float[]> legacyTrace = new LinkedHashMap<>();
        try (LlamaModel tensorRef = loadLlamaModel(modelDir, true);
             LlamaModel legacy = loadLlamaModel(modelDir, false)) {
            tensorRef.setLayerDebugHook(event -> captureTrace(tensorRefTrace, event));
            legacy.setLayerDebugHook(event -> captureTrace(legacyTrace, event));
            try (AbstractTensor ignoredRef = tensorRef.batchForward(hfSampleTokenIds(), 0);
                 AbstractTensor ignoredLegacy = legacy.batchForward(hfSampleTokenIds(), 0)) {
                assertTrue(legacyTrace.keySet().stream().allMatch(tensorRefTrace::containsKey),
                        "TensorRef trace is missing legacy stages: " + missingStages(legacyTrace, tensorRefTrace));
                for (String key : legacyTrace.keySet()) {
                    float[] expected = legacyTrace.get(key);
                    float[] actual = tensorRefTrace.get(key);
                    assertEquals(expected.length, actual.length, key + " length");
                    for (int i = 0; i < expected.length; i++) {
                        assertEquals(expected[i], actual[i], 1.0e-3f, key + " index=" + i);
                    }
                }
            }
        }
    }

    @Test
    public void tinyTensorRefCachedKvMatchesColdReplay() {
        Path modelDir = writeTinyCheckpoint(tempDir.resolve("llama-tiny-kv-parity"), tinyConfig(), 202);
        int[] prompt = hfSampleTokenIds();
        int continuation = 7;
        int[] replay = Arrays.copyOf(prompt, prompt.length + 1);
        replay[prompt.length] = continuation;
        try (LlamaModel model = loadLlamaModel(modelDir, true);
             KvCacheSession cached = model.newKvCacheSession();
             KvCacheSession cold = model.newKvCacheSession();
             KvCacheSession coldReplaySession = model.newKvCacheSession();
             AbstractTensor cachedPrompt = model.batchForward(prompt, 0, cached);
             AbstractTensor coldPrompt = model.batchForward(prompt, 0, cold)) {
            assertPrefillTensorsClose(coldPrompt, cachedPrompt, 1.0e-3f, "cached/cold prefill");
            try (AbstractTensor cachedDecode = model.forward(continuation, prompt.length, cached);
                 AbstractTensor coldReplay = model.batchForward(replay, 0, coldReplaySession)) {
                assertEquals(coldReplay.shape().last(), cachedDecode.shape().last());
                int lastRow = coldReplay.shape().first() - 1;
                for (int column = 0; column < cachedDecode.shape().last(); column++) {
                    assertEquals(coldReplay.get(lastRow, column), cachedDecode.get(0, column), 1.0e-3f,
                            "cached/cold decode column=" + column);
                }
            }
        }
    }

    @Test
    public void tiedOutputHeadQuantizationKeepsInputEmbeddingsUsable() {
        Path modelDir = writeTinyCheckpoint(tempDir.resolve("llama-tiny-tied-head"), tinyConfig(), 203);
        assertDoesNotThrow(() -> {
            try (LlamaModel model = loadLlamaModel(modelDir, true, Optional.of(DType.Q4));
                 KvCacheSession session = model.newKvCacheSession();
                 TensorRef output = model.batchForwardRef(hfSampleTokenIds(), 0, session)) {
                assertEquals(tinyConfig().embeddingLength, output.shape().last());
            }
        }, "quantizing a tied output head must not close the shared embedding table");
    }

    @Test
    public void tensorRefLlamaRetainsRequiredLoraHotSwapCapability() {
        Path modelDir = writeTinyCheckpoint(tempDir.resolve("llama-tiny-lora-capability"), tinyConfig(), 204);
        try (LlamaModel model = loadLlamaModel(modelDir, true)) {
            assertTrue(model.supportsLoraHotSwap(),
                    "Llama migration is incomplete until TensorRef preserves LoRA hot-swap");
        }
    }

    @Test
    public void tensorRefLlamaAppliesAndClearsLoraHotSwap() throws Exception {
        Path modelDir = writeTinyCheckpoint(tempDir.resolve("llama-tiny-lora-apply"), tinyConfig(), 211);
        Path adapterDir = writeTinyLoraAdapter(tempDir.resolve("llama-tiny-lora-adapter"), tinyConfig());
        int[] tokens = hfSampleTokenIds();
        try (LlamaModel model = loadLlamaModel(modelDir, true);
             LoraAdapter adapter = LoraAdapter.load(adapterDir.toFile())) {
            model.registerLoraAdapter("test", adapter);
            TensorRef base;
            try (KvCacheSession session = model.newKvCacheSession()) {
                base = model.batchForwardRef(tokens, 0, session);
            }
            model.setActiveAdapter("test");
            TensorRef adapted;
            try (KvCacheSession session = model.newKvCacheSession()) {
                adapted = model.batchForwardRef(tokens, 0, session);
            }
            model.clearActiveAdapter();
            TensorRef cleared;
            try (KvCacheSession session = model.newKvCacheSession()) {
                cleared = model.batchForwardRef(tokens, 0, session);
            }
            try (base; adapted; cleared) {
                assertTrue(maxDifference(base, adapted) > 1.0e-4f,
                        "activating LoRA must change the TensorRef output");
                assertEquals(0.0f, maxDifference(base, cleared), 1.0e-3f,
                        "clearing LoRA must restore the base TensorRef output");
            }
        }
    }

    @Test
    public void tensorParallelLlamaIsRejectedBeforeWeightsAreLoaded() {
        LlamaConfig tensorParallelConfig = new LlamaConfig(32, 32, 64, 2, 2, 2, 1.0e-6f,
                32, 1, 2, ActivationFunction.Type.SILU, 10_000.0, null);
        Path modelDir = writeTinyCheckpoint(tempDir.resolve("llama-tiny-tp-rejection"),
                tensorParallelConfig, 205);
        MetricRegistry metrics = new MetricRegistry();
        TensorAllocator allocator = new ArrayQueueTensorAllocator(metrics);
        WrappedForkJoinPool pool = new WrappedForkJoinPool(new ForkJoinPool(1));
        LlamaModel model = createLlamaModel(modelDir, true, Optional.empty(),
                new StaticTensorParallelContext(0, 2), Mockito.mock(TensorParallelCollectives.class),
                new ConfigurableTensorProvider(new NaiveTensorOperations()), metrics, allocator, pool);
        try {
            assertThrows(UnsupportedOperationException.class, model::init,
                    "incomplete TensorRef tensor parallelism must fail during initialization, not inference");
        } finally {
            model.close();
        }
    }

    @Test
    public void genuineQ4CheckpointMatchesLegacyOnPanama() {
        Path modelDir = writeTinyQ4Checkpoint(tempDir.resolve("llama-tiny-q4-panama"), tinyConfig(), 206);
        MetricRegistry refMetrics = new MetricRegistry();
        TensorAllocator refAllocator = new ArrayQueueTensorAllocator(refMetrics);
        WrappedForkJoinPool refPool = new WrappedForkJoinPool(new ForkJoinPool(2));
        MetricRegistry legacyMetrics = new MetricRegistry();
        TensorAllocator legacyAllocator = new ArrayQueueTensorAllocator(legacyMetrics);
        WrappedForkJoinPool legacyPool = new WrappedForkJoinPool(new ForkJoinPool(2));
        try (LlamaModel tensorRef = loadLlamaModel(modelDir, true, Optional.empty(),
                     new StaticTensorParallelContext(0, 1), new SingleRankTensorParallelCollectives(),
                     new ConfigurableTensorProvider(new PanamaTensorOperations(MachineSpec.VECTOR_TYPE,
                             refAllocator, refPool)), refMetrics, refAllocator, refPool);
             LlamaModel legacy = loadLlamaModel(modelDir, false, Optional.empty(),
                     new StaticTensorParallelContext(0, 1), new SingleRankTensorParallelCollectives(),
                     new ConfigurableTensorProvider(new PanamaTensorOperations(MachineSpec.VECTOR_TYPE,
                             legacyAllocator, legacyPool)), legacyMetrics, legacyAllocator, legacyPool);
             AbstractTensor refOutput = tensorRef.batchForward(hfSampleTokenIds(), 0);
             AbstractTensor legacyOutput = legacy.batchForward(hfSampleTokenIds(), 0)) {
            assertPrefillTensorsClose(legacyOutput, refOutput, 1.0e-3f, "Q4 Panama legacy/TensorRef prefill");
        }
    }

    @Test
    public void genuineQ4NativeAndPanamaMatchAcrossPrefillAndDecodeTokens() {
        Assumptions.assumeTrue(NativeOps.isAvailable(), "Tensor2 native library is unavailable");
        Path modelDir = writeTinyQ4Checkpoint(tempDir.resolve("llama-tiny-q4-native-panama"), tinyConfig(), 210);
        MetricRegistry nativeMetrics = new MetricRegistry();
        TensorAllocator nativeAllocator = new ArrayQueueTensorAllocator(nativeMetrics);
        WrappedForkJoinPool nativePool = new WrappedForkJoinPool(new ForkJoinPool(2));
        MetricRegistry panamaMetrics = new MetricRegistry();
        TensorAllocator panamaAllocator = new ArrayQueueTensorAllocator(panamaMetrics);
        WrappedForkJoinPool panamaPool = new WrappedForkJoinPool(new ForkJoinPool(2));
        LlamaModel nativeModel = createLlamaModel(modelDir, true, Optional.empty(),
                new StaticTensorParallelContext(0, 1), new SingleRankTensorParallelCollectives(),
                new ConfigurableTensorProvider(new PanamaTensorOperations(MachineSpec.VECTOR_TYPE,
                        nativeAllocator, nativePool)), nativeMetrics, nativeAllocator, nativePool);
        LlamaModel panamaModel = createLlamaModel(modelDir, true, Optional.empty(),
                new StaticTensorParallelContext(0, 1), new SingleRankTensorParallelCollectives(),
                new ConfigurableTensorProvider(new PanamaTensorOperations(MachineSpec.VECTOR_TYPE,
                        panamaAllocator, panamaPool)), panamaMetrics, panamaAllocator, panamaPool);
        nativeModel.getLighter().putTensorOperations(TensorProviderKind.SIMD, new NativeOps());
        panamaModel.getLighter().putTensorOperations(TensorProviderKind.SIMD,
                panamaModel.getLighter().tensorOperations().get(TensorProviderKind.PANAMA));
        nativeModel.init();
        panamaModel.init();

        Map<String, float[]> nativeTrace = new LinkedHashMap<>();
        Map<String, float[]> panamaTrace = new LinkedHashMap<>();
        nativeModel.setLayerDebugHook(event -> captureTrace(nativeTrace, event));
        panamaModel.setLayerDebugHook(event -> captureTrace(panamaTrace, event));
        GenerationEngineRef sampler = new GenerationEngineRef();
        GeneratorParameters parameters = new GeneratorParameters().withTemperature(0.0f);
        int[] prompt = hfSampleTokenIds();

        try (nativeModel; panamaModel;
             KvCacheSession nativeSession = nativeModel.newKvCacheSession();
             KvCacheSession panamaSession = panamaModel.newKvCacheSession();
             TensorRef nativeLogits = sampler.allocateLogits(nativeModel);
             TensorRef panamaLogits = sampler.allocateLogits(panamaModel);
             TensorRef nativeScratch = sampler.allocateArgMaxScratch(nativeModel);
             TensorRef panamaScratch = sampler.allocateArgMaxScratch(panamaModel)) {
            int next;
            try (TensorRef nativeOutput = nativeModel.batchForwardRef(prompt, 0, nativeSession);
                 TensorRef panamaOutput = panamaModel.batchForwardRef(prompt, 0, panamaSession)) {
                next = assertNativePanamaStep("prefill", nativeModel, panamaModel, nativeOutput, panamaOutput,
                        nativeTrace, panamaTrace, sampler, parameters, nativeLogits, panamaLogits,
                        nativeScratch, panamaScratch);
            }

            for (int position = prompt.length; position < prompt.length + 4; position++) {
                nativeTrace.clear();
                panamaTrace.clear();
                try (TensorRef nativeOutput = nativeModel.forwardRef(next, position, nativeSession, Optional.empty());
                     TensorRef panamaOutput = panamaModel.forwardRef(next, position, panamaSession, Optional.empty())) {
                    next = assertNativePanamaStep("decode position=" + position, nativeModel, panamaModel,
                            nativeOutput, panamaOutput, nativeTrace, panamaTrace, sampler, parameters,
                            nativeLogits, panamaLogits, nativeScratch, panamaScratch);
                }
            }
        }
    }

    private static int assertNativePanamaStep(String label, LlamaModel nativeModel, LlamaModel panamaModel,
            TensorRef nativeOutput, TensorRef panamaOutput, Map<String, float[]> nativeTrace,
            Map<String, float[]> panamaTrace, GenerationEngineRef sampler, GeneratorParameters parameters,
            TensorRef nativeLogits, TensorRef panamaLogits, TensorRef nativeScratch, TensorRef panamaScratch) {
        assertEquals(panamaTrace.keySet(), nativeTrace.keySet(), label + " stages");
        for (String stage : panamaTrace.keySet()) {
            float[] expected = panamaTrace.get(stage);
            float[] actual = nativeTrace.get(stage);
            assertEquals(expected.length, actual.length, label + " " + stage + " length");
            for (int index = 0; index < expected.length; index++) {
                assertEquals(expected[index], actual[index], 1.0e-3f,
                        label + " " + stage + " index=" + index);
            }
        }
        assertTensorRefsClose(panamaOutput, nativeOutput, 1.0e-3f, label + " hidden");

        SamplerReturn nativeSample = sampler.sample(nativeModel, parameters, nativeOutput, nativeLogits,
                nativeScratch, new ResponseContext(nativeModel), new Random(1));
        SamplerReturn panamaSample = sampler.sample(panamaModel, parameters, panamaOutput, panamaLogits,
                panamaScratch, new ResponseContext(panamaModel), new Random(1));
        assertTensorRefsClose(panamaLogits, nativeLogits, 1.0e-3f, label + " logits");
        assertEquals(panamaSample.getToken(), nativeSample.getToken(), label + " greedy token");
        return nativeSample.getToken();
    }

    private static void assertTensorRefsClose(TensorRef expected, TensorRef actual, float tolerance, String label) {
        assertEquals(expected.shape(), actual.shape(), label + " shape");
        for (int row = 0; row < expected.shape().first(); row++) {
            for (int column = 0; column < expected.shape().last(); column++) {
                assertEquals(expected.get(row, column), actual.get(row, column), tolerance,
                        label + " row=" + row + " column=" + column);
            }
        }
    }

    @Test
    public void llamaSubclassesDoNotInheritConcreteTensorRefOrKv2OptIn() {
        Path modelDir = writeTinyCheckpoint(tempDir.resolve("llama-tiny-subclass-scope"), tinyConfig(), 207);
        try (LlamaModel subclass = loadLlamaModel(modelDir, false)) {
            assertFalse(subclass.usesTensorRefExecution());
            assertFalse(subclass.usesKvCache2Generation());
            assertThrows(UnsupportedOperationException.class, subclass::loadTransformerBlockWeights2);
        }
    }

    @Test
    public void tensorRefForwardDoesNotUseLegacyTensorAllocatorMeters() {
        Path modelDir = writeTinyCheckpoint(tempDir.resolve("llama-tiny-allocator-boundary"), tinyConfig(), 208);
        MetricRegistry metrics = new MetricRegistry();
        TensorAllocator allocator = new ArrayQueueTensorAllocator(metrics);
        WrappedForkJoinPool pool = new WrappedForkJoinPool(new ForkJoinPool(2));
        try (LlamaModel model = loadLlamaModel(modelDir, true, Optional.empty(),
                     new StaticTensorParallelContext(0, 1), new SingleRankTensorParallelCollectives(),
                     new ConfigurableTensorProvider(new NaiveTensorOperations()), metrics, allocator, pool)) {
            long cleanBefore = metrics.meter("tensorcache.get").getCount();
            long dirtyBefore = metrics.meter("tensorcache.dirtyget").getCount();
            try (KvCacheSession session = model.newKvCacheSession();
                 TensorRef ignored = model.batchForwardRef(hfSampleTokenIds(), 0, session)) {
                assertEquals(cleanBefore, metrics.meter("tensorcache.get").getCount());
                assertEquals(dirtyBefore, metrics.meter("tensorcache.dirtyget").getCount());
            }
        }
    }

    public static LlamaConfig tinyConfig() {
        return new LlamaConfig(32, 32, 64, 2, 1, 2, 1.0e-6f, 32, 1, 2,
                ActivationFunction.Type.SILU, 10_000.0, null);
    }

    private static Map<String, Object> tinyConfigJson(LlamaConfig config) {
        Map<String, Object> json = new LinkedHashMap<>();
        json.put("model_type", "llama");
        json.put("architectures", List.of("LlamaForCausalLM"));
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
        return json;
    }

    public static Path writeTinyCheckpoint(Path dir, LlamaConfig config, int seed) {
        return writeTinyCheckpoint(dir, config, seed, false);
    }

    static Path writeTinyQ4Checkpoint(Path dir, LlamaConfig config, int seed) {
        return writeTinyCheckpoint(dir, config, seed, true);
    }

    private static Path writeTinyCheckpoint(Path dir, LlamaConfig config, int seed, boolean q4) {
        try {
            Files.createDirectories(dir);
            JsonUtils.om.writeValue(dir.resolve("config.json").toFile(), tinyConfigJson(config));
            Map<String, AbstractTensor> tensors = new LinkedHashMap<>();
            putMatrix(tensors, "model.embed_tokens.weight",
                    matrix(config.vocabularySize, config.embeddingLength, seed++), q4);
            tensors.put("model.norm.weight", ones(1, config.embeddingLength));
            for (int i = 0; i < config.numberOfLayers; i++) {
                String layer = "model.layers." + i + ".";
                tensors.put(layer + "input_layernorm.weight", ones(1, config.embeddingLength));
                tensors.put(layer + "post_attention_layernorm.weight", ones(1, config.embeddingLength));
                putMatrix(tensors, layer + "self_attn.q_proj.weight",
                        matrix(config.attentionLength, config.embeddingLength, seed++), q4);
                putMatrix(tensors, layer + "self_attn.k_proj.weight",
                        matrix(config.kvLength, config.embeddingLength, seed++), q4);
                putMatrix(tensors, layer + "self_attn.v_proj.weight",
                        matrix(config.kvLength, config.embeddingLength, seed++), q4);
                putMatrix(tensors, layer + "self_attn.o_proj.weight",
                        matrix(config.embeddingLength, config.attentionLength, seed++), q4);
                putMatrix(tensors, layer + "mlp.gate_proj.weight",
                        matrix(config.hiddenLength, config.embeddingLength, seed++), q4);
                putMatrix(tensors, layer + "mlp.up_proj.weight",
                        matrix(config.hiddenLength, config.embeddingLength, seed++), q4);
                putMatrix(tensors, layer + "mlp.down_proj.weight",
                        matrix(config.embeddingLength, config.hiddenLength, seed++), q4);
            }
            SafeTensorWriter.writeModel(dir, Map.of("format", "pt"), tensors, 1 << 28);
            tensors.values().forEach(AbstractTensor::close);
            return dir;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static void putMatrix(Map<String, AbstractTensor> tensors, String name, FloatBufferTensor matrix,
            boolean q4) {
        if (!q4) {
            tensors.put(name, matrix);
            return;
        }
        tensors.put(name, new Q4ByteBufferTensor(matrix));
        matrix.close();
    }

    public static LlamaModel loadLlamaModel(Path modelDir, boolean tensorRefExecution) {
        return loadLlamaModel(modelDir, tensorRefExecution, Optional.empty());
    }

    static LlamaModel loadLlamaModel(Path modelDir, boolean tensorRefExecution,
            Optional<DType> outputHeadQuantization) {
        MetricRegistry metrics = new MetricRegistry();
        TensorAllocator allocator = new ArrayQueueTensorAllocator(metrics);
        WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores());
        return loadLlamaModel(modelDir, tensorRefExecution, outputHeadQuantization,
                new StaticTensorParallelContext(0, 1), new SingleRankTensorParallelCollectives(),
                new ConfigurableTensorProvider(new NaiveTensorOperations()), metrics, allocator, pool);
    }

    private static LlamaModel loadLlamaModel(Path modelDir, boolean tensorRefExecution,
            Optional<DType> outputHeadQuantization, TensorParallelContext tensorParallelContext,
            TensorParallelCollectives tensorParallelCollectives, ConfigurableTensorProvider provider,
            MetricRegistry metrics, TensorAllocator allocator, WrappedForkJoinPool pool) {
        LlamaModel model = createLlamaModel(modelDir, tensorRefExecution, outputHeadQuantization,
                tensorParallelContext, tensorParallelCollectives, provider, metrics, allocator, pool);
        if (provider.get() instanceof NaiveTensorOperations) {
            model.getLighter().putTensorOperations(TensorProviderKind.SIMD,
                    model.getLighter().tensorOperations().get(TensorProviderKind.NAIVE));
        } else if (provider.get() instanceof PanamaTensorOperations) {
            model.getLighter().putTensorOperations(TensorProviderKind.SIMD,
                    model.getLighter().tensorOperations().get(TensorProviderKind.PANAMA));
        }
        model.init();
        // AbstractModel installs optional native TensorRef ops during construction. These
        // characterization models deliberately select their provider explicitly so legacy and
        // TensorRef traces are compared on the same arithmetic path.
        if (provider.get() instanceof NaiveTensorOperations) {
            model.getLighter().putTensorOperations(TensorProviderKind.SIMD,
                    model.getLighter().tensorOperations().get(TensorProviderKind.NAIVE));
        } else if (provider.get() instanceof PanamaTensorOperations) {
            model.getLighter().putTensorOperations(TensorProviderKind.SIMD,
                    model.getLighter().tensorOperations().get(TensorProviderKind.PANAMA));
        }
        return model;
    }

    private static LlamaModel createLlamaModel(Path modelDir, boolean tensorRefExecution,
            Optional<DType> outputHeadQuantization, TensorParallelContext tensorParallelContext,
            TensorParallelCollectives tensorParallelCollectives, ConfigurableTensorProvider provider,
            MetricRegistry metrics, TensorAllocator allocator, WrappedForkJoinPool pool) {
        LlamaModel model;
        if (tensorRefExecution) {
            // LlamaModel's TensorRef loader deliberately requires the concrete class.
            model = new LlamaModel(AbstractModel.InferenceType.FULL_GENERATION,
                    configFromFile(modelDir), new DefaultWeightLoader(modelDir.toFile()),
                    Mockito.mock(PreTrainedTokenizer.class), DType.F32, DType.I8, Optional.of(DType.Q4),
                    provider, metrics, allocator,
                    new KvBufferCacheSettings(true), new DefaultToolCallParser(), pool,
                    tensorParallelContext, tensorParallelCollectives, outputHeadQuantization);
        } else {
            model = new LlamaModel(AbstractModel.InferenceType.FULL_GENERATION,
                    configFromFile(modelDir), new DefaultWeightLoader(modelDir.toFile()),
                    Mockito.mock(PreTrainedTokenizer.class), DType.F32, DType.I8, Optional.of(DType.Q4),
                    provider, metrics, allocator,
                    new KvBufferCacheSettings(true), new DefaultToolCallParser(), pool,
                    tensorParallelContext, tensorParallelCollectives, outputHeadQuantization) {
                @Override
                protected boolean usesTensorRefExecution() {
                    return false;
                }
            };
        }
        return model;
    }

    private static LlamaConfig configFromFile(Path modelDir) {
        try {
            return JsonUtils.om.readValue(modelDir.resolve("config.json").toFile(), LlamaConfig.class);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static FloatBufferTensor matrix(int rows, int cols, int seed) {
        FloatBufferTensor tensor = new FloatBufferTensor(rows, cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                tensor.set(((row * 13 + col * 7 + seed) % 17 - 8) / 8.0f, row, col);
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

    private static Path writeTinyLoraAdapter(Path dir, LlamaConfig config) {
        try {
            Files.createDirectories(dir);
            Files.writeString(dir.resolve(LoraAdapterConfig.FILE_NAME),
                    "{\"r\":1,\"lora_alpha\":1.0,\"target_modules\":[\"q_proj\",\"k_proj\",\"v_proj\",\"o_proj\",\"gate_proj\",\"up_proj\",\"down_proj\"]}");
            Map<String, AbstractTensor> tensors = new LinkedHashMap<>();
            for (int layer = 0; layer < config.numberOfLayers; layer++) {
                String base = "model.layers." + layer + ".";
                int seed = 301 + layer * 10;
                addTinyLora(tensors, base + "self_attn.q_proj", config.attentionLength, config.embeddingLength, seed++);
                addTinyLora(tensors, base + "self_attn.k_proj", config.kvLength, config.embeddingLength, seed++);
                addTinyLora(tensors, base + "self_attn.v_proj", config.kvLength, config.embeddingLength, seed++);
                addTinyLora(tensors, base + "self_attn.o_proj", config.embeddingLength, config.attentionLength, seed++);
                addTinyLora(tensors, base + "mlp.gate_proj", config.hiddenLength, config.embeddingLength, seed++);
                addTinyLora(tensors, base + "mlp.up_proj", config.hiddenLength, config.embeddingLength, seed++);
                addTinyLora(tensors, base + "mlp.down_proj", config.embeddingLength, config.hiddenLength, seed);
            }
            SafeTensorWriter.write(dir.resolve(LoraAdapter.SAFETENSORS_FILE_NAME), Map.of(), tensors);
            tensors.values().forEach(AbstractTensor::close);
            return dir;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static void addTinyLora(Map<String, AbstractTensor> tensors, String baseName,
            int outputLength, int inputLength, int seed) {
        String prefix = "base_model.model." + baseName;
        tensors.put(prefix + ".lora_A.weight", matrix(1, inputLength, seed));
        tensors.put(prefix + ".lora_B.weight", matrix(outputLength, 1, seed + 1));
    }

    private static float maxDifference(TensorRef first, TensorRef second) {
        float max = 0.0f;
        for (int row = 0; row < first.shape().first(); row++) {
            for (int column = 0; column < first.shape().last(); column++) {
                max = Math.max(max, Math.abs(first.get(row, column) - second.get(row, column)));
            }
        }
        return max;
    }

    private static void assertShape(Map<String, TensorInfo> info, String name, int... expected) {
        assertTrue(info.containsKey(name), "missing tensor " + name);
        assertTrue(Arrays.equals(expected, info.get(name).shape), name + " shape");
    }

    private static void captureTrace(Map<String, float[]> trace, AbstractModel.LayerDebugEvent event) {
        AbstractTensor tensor = event.hiddenStates();
        float[] values = new float[Math.toIntExact(tensor.shape().size())];
        int index = 0;
        for (int row = 0; row < tensor.shape().first(); row++) {
            for (int column = 0; column < tensor.shape().last(); column++) {
                values[index++] = tensor.get(row, column);
            }
        }
        trace.put(event.layerIndex() + ":" + event.stage(), values);
    }

    private static List<String> missingStages(Map<String, float[]> expected, Map<String, float[]> actual) {
        return expected.keySet().stream().filter(key -> !actual.containsKey(key)).toList();
    }

    private static void assertPrefillTensorsClose(AbstractTensor expected, AbstractTensor actual, float tolerance,
                                                   String label) {
        assertEquals(expected.shape(), actual.shape(), label + " shape");
        float maxAbs = 0.0f;
        for (int row = 0; row < expected.shape().first(); row++) {
            for (int col = 0; col < expected.shape().last(); col++) {
                maxAbs = Math.max(maxAbs, Math.abs(expected.get(row, col) - actual.get(row, col)));
            }
        }
        assertTrue(maxAbs <= tolerance, label + " maxAbs=" + maxAbs);
    }
}
