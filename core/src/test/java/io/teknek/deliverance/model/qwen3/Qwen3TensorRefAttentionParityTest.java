package io.teknek.deliverance.model.qwen3;

import io.teknek.deliverance.JsonUtils;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.generator2.KvCacheSelfAttention2;
import io.teknek.deliverance.generator2.TransformerBlock2;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.kv.KvCacheSession;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorOps;
import io.teknek.deliverance.tensor2.TensorProviderKind;
import io.teknek.deliverance.tensor2.TensorRef;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.io.BufferedWriter;
import java.io.IOException;
import java.lang.reflect.Field;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Focused TensorRef attention and KV-cache parity tests. */
class Qwen3TensorRefAttentionParityTest {
    @TempDir
    Path tempDir;

    @Test
    void cachedDecodeAttentionValueMatchesColdReplay() {
        Path modelDir = Qwen3HfTextModelPortedTest.writeTinyCheckpoint(
                tempDir.resolve("qwen3-tiny-cached-attention-value"),
                Qwen3HfTextModelPortedTest.tinyConfig(), 12_346);
        int[] prompt = new int[]{3, 4, 5, 6};
        int continuation = 7;
        int[] replayTokens = Arrays.copyOf(prompt, prompt.length + 1);
        replayTokens[prompt.length] = continuation;
        Map<String, float[]> cachedStages = new LinkedHashMap<>();
        Map<String, float[]> coldStages = new LinkedHashMap<>();
        TraceRecorder cachedTrace = new TraceRecorder("cached");
        TraceRecorder coldTrace = new TraceRecorder("cold");

        try (Qwen3Model model = Qwen3HfTextModelPortedTest.loadTinyQwen3Model(modelDir);
             KvCacheSession cachedSession = model.newKvCacheSession();
             KvCacheSession coldReplaySession = model.newKvCacheSession()) {
            model.setLayerDebugHook(event -> {
                cachedTrace.record(event);
                if (event.layerIndex() == 0 && isAttentionStage(event.stage())) {
                    cachedStages.put(event.stage(), lastRowValues(event.hiddenStates()));
                }
            });
            try (AbstractTensor promptOutput = model.batchForward(prompt, 0, cachedSession)) {
                cachedTrace.clear();
            }
            try (AbstractTensor cachedDecode = model.forward(continuation, prompt.length, cachedSession)) {
                assertFinite(cachedDecode);
            }

            model.setLayerDebugHook(event -> {
                coldTrace.record(event);
                if (event.layerIndex() == 0 && isAttentionStage(event.stage())) {
                    coldStages.put(event.stage(), lastRowValues(event.hiddenStates()));
                }
            });
            try (AbstractTensor coldReplay = model.batchForward(replayTokens, 0, coldReplaySession)) {
                assertFinite(coldReplay);
            }
        }

        writeTracesIfRequested(cachedTrace, coldTrace);
        compareTraces(cachedTrace.snapshots, coldTrace.snapshots);

        assertEquals(coldStages.keySet(), cachedStages.keySet(),
                "cached and cold attention stages");
        for (String stage : cachedStages.keySet()) {
            float[] cached = cachedStages.get(stage);
            float[] cold = coldStages.get(stage);
            assertEquals(cold.length, cached.length, "stage=" + stage + " attention value width");
            float maxAbs = 0.0f;
            for (int column = 0; column < cached.length; column++) {
                maxAbs = Math.max(maxAbs, Math.abs(cached[column] - cold[column]));
            }
            assertTrue(maxAbs <= 1.0e-4f,
                    "stage=" + stage + " cached versus cold maxAbs=" + maxAbs);
        }
    }

    @Test
    void legacyCachedDecodeMatchesColdReplay() {
        Path modelDir = Qwen3HfTextModelPortedTest.writeTinyCheckpoint(
                tempDir.resolve("qwen3-legacy-cached-replay"),
                Qwen3HfTextModelPortedTest.tinyConfig(), 12_346);
        int[] prompt = new int[]{3, 4, 5, 6};
        int continuation = 7;
        int[] replayTokens = Arrays.copyOf(prompt, prompt.length + 1);
        replayTokens[prompt.length] = continuation;

        try (Qwen3Model model = Qwen3HfTextModelPortedTest.loadTinyQwen3Model(modelDir, false);
             KvCacheSession cachedSession = model.newKvCacheSession();
             KvCacheSession coldSession = model.newKvCacheSession()) {
            try (AbstractTensor promptOutput = model.batchForward(prompt, 0, cachedSession);
                 AbstractTensor cachedDecode = model.forward(continuation, prompt.length, cachedSession);
                 AbstractTensor coldReplay = model.batchForward(replayTokens, 0, coldSession)) {
                int lastRow = coldReplay.shape().first() - 1;
                float maxAbs = 0.0f;
                for (int column = 0; column < coldReplay.shape().last(); column++) {
                    maxAbs = Math.max(maxAbs,
                            Math.abs(cachedDecode.get(0, column) - coldReplay.get(lastRow, column)));
                }
                assertTrue(maxAbs <= 1.0e-4f, "legacy cached versus cold maxAbs=" + maxAbs);
            }
        }
    }

    @Test
    void legacyCachedDecodeTraceMatchesColdReplay() {
        Path modelDir = Qwen3HfTextModelPortedTest.writeTinyCheckpoint(
                tempDir.resolve("qwen3-legacy-trace-replay"),
                Qwen3HfTextModelPortedTest.tinyConfig(), 12_346);
        int[] prompt = new int[]{3, 4, 5, 6};
        int continuation = 7;
        int[] replayTokens = Arrays.copyOf(prompt, prompt.length + 1);
        replayTokens[prompt.length] = continuation;
        TraceRecorder cachedTrace = new TraceRecorder("legacy-cached");
        TraceRecorder coldTrace = new TraceRecorder("legacy-cold");

        try (Qwen3Model model = Qwen3HfTextModelPortedTest.loadTinyQwen3Model(modelDir, false);
             KvCacheSession cachedSession = model.newKvCacheSession();
             KvCacheSession coldSession = model.newKvCacheSession()) {
            model.setLayerDebugHook(cachedTrace::record);
            try (AbstractTensor ignored = model.batchForward(prompt, 0, cachedSession)) {
                cachedTrace.clear();
            }
            try (AbstractTensor ignored = model.forward(continuation, prompt.length, cachedSession)) {
                // capture decode trace
            }
            model.setLayerDebugHook(coldTrace::record);
            try (AbstractTensor ignored = model.batchForward(replayTokens, 0, coldSession)) {
                // capture cold replay trace
            }
        }

        compareTraces(cachedTrace.snapshots, coldTrace.snapshots);
    }

    @Test
    void legacyAndTensorRefCachedColdCommonStagesMatch() {
        Path modelDir = Qwen3HfTextModelPortedTest.writeTinyCheckpoint(
                tempDir.resolve("qwen3-legacy-tensorref-cached-cold"),
                Qwen3HfTextModelPortedTest.tinyConfig(), 12_346);
        int[] prompt = new int[]{3, 4, 5, 6};
        int continuation = 7;
        int[] replayTokens = Arrays.copyOf(prompt, prompt.length + 1);
        replayTokens[prompt.length] = continuation;

        TraceRecorder tensorRefCached = new TraceRecorder("tensorref-cached");
        TraceRecorder tensorRefCold = new TraceRecorder("tensorref-cold");
        TraceRecorder legacyCached = new TraceRecorder("legacy-cached");
        TraceRecorder legacyCold = new TraceRecorder("legacy-cold");

        try (Qwen3Model tensorRef = Qwen3HfTextModelPortedTest.loadTinyQwen3Model(modelDir);
             Qwen3Model legacy = Qwen3HfTextModelPortedTest.loadTinyQwen3Model(modelDir, false);
             KvCacheSession tensorRefCachedSession = tensorRef.newKvCacheSession();
             KvCacheSession tensorRefColdSession = tensorRef.newKvCacheSession();
             KvCacheSession legacyCachedSession = legacy.newKvCacheSession();
             KvCacheSession legacyColdSession = legacy.newKvCacheSession()) {
            tensorRef.setLayerDebugHook(tensorRefCached::record);
            legacy.setLayerDebugHook(legacyCached::record);
            try (AbstractTensor ignored = tensorRef.batchForward(prompt, 0, tensorRefCachedSession);
                 AbstractTensor ignoredLegacy = legacy.batchForward(prompt, 0, legacyCachedSession)) {
                tensorRefCached.clear();
                legacyCached.clear();
            }
            try (AbstractTensor ignored = tensorRef.forward(continuation, prompt.length, tensorRefCachedSession);
                 AbstractTensor ignoredLegacy = legacy.forward(continuation, prompt.length, legacyCachedSession)) {
                // capture cached decode traces
            }

            tensorRef.setLayerDebugHook(tensorRefCold::record);
            legacy.setLayerDebugHook(legacyCold::record);
            try (AbstractTensor ignored = tensorRef.batchForward(replayTokens, 0, tensorRefColdSession);
                 AbstractTensor ignoredLegacy = legacy.batchForward(replayTokens, 0, legacyColdSession)) {
                // capture cold replay traces
            }
        }

        compareCommonTraces(tensorRefCached.snapshots, legacyCached.snapshots, "cached");
        compareCommonTraces(tensorRefCold.snapshots, legacyCold.snapshots, "cold");
    }

    @Test
    void tensorRefLayer1CachedColdTrace() {
        Path modelDir = Qwen3HfTextModelPortedTest.writeTinyCheckpoint(
                tempDir.resolve("qwen3-layer1-cached-cold"),
                Qwen3HfTextModelPortedTest.tinyConfig(), 12_346);
        int[] prompt = new int[]{3, 4, 5, 6};
        int continuation = 7;
        int[] replayTokens = Arrays.copyOf(prompt, prompt.length + 1);
        replayTokens[prompt.length] = continuation;
        TraceRecorder cachedTrace = new TraceRecorder("tensorref-layer1-cached");
        TraceRecorder coldTrace = new TraceRecorder("tensorref-layer1-cold");

        try (Qwen3Model model = Qwen3HfTextModelPortedTest.loadTinyQwen3Model(modelDir);
             KvCacheSession cachedSession = model.newKvCacheSession();
             KvCacheSession coldSession = model.newKvCacheSession()) {
            model.setLayerDebugHook(cachedTrace::record);
            try (AbstractTensor ignored = model.batchForward(prompt, 0, cachedSession)) {
                cachedTrace.clear();
            }
            try (AbstractTensor ignored = model.forward(continuation, prompt.length, cachedSession)) {
                // capture cached layer-1 decode
            }
            model.setLayerDebugHook(coldTrace::record);
            try (AbstractTensor ignored = model.batchForward(replayTokens, 0, coldSession)) {
                // capture cold layer-1 replay
            }
        }

        compareTraces(layerSnapshots(cachedTrace.snapshots, 1), layerSnapshots(coldTrace.snapshots, 1));
    }

    @ParameterizedTest(name = "layer1 cached/cold with {0}")
    @ValueSource(strings = {"PANAMA", "NAIVE"})
    void tensorRefLayer1CachedColdTraceWithProvider(String providerName) {
        Path modelDir = Qwen3HfTextModelPortedTest.writeTinyCheckpoint(
                tempDir.resolve("qwen3-layer1-provider-" + providerName),
                Qwen3HfTextModelPortedTest.tinyConfig(), 12_346);
        int[] prompt = new int[]{3, 4, 5, 6};
        int continuation = 7;
        int[] replayTokens = Arrays.copyOf(prompt, prompt.length + 1);
        replayTokens[prompt.length] = continuation;
        TraceRecorder cachedTrace = new TraceRecorder("tensorref-provider-cached");
        TraceRecorder coldTrace = new TraceRecorder("tensorref-provider-cold");

        try (Qwen3Model model = Qwen3HfTextModelPortedTest.loadTinyQwen3Model(modelDir);
             KvCacheSession cachedSession = model.newKvCacheSession();
             KvCacheSession coldSession = model.newKvCacheSession()) {
            TensorProviderKind kind = TensorProviderKind.valueOf(providerName);
            model.getLighter().putTensorOperations(TensorProviderKind.SIMD,
                    model.getLighter().tensorOperations().get(kind));
            model.setLayerDebugHook(cachedTrace::record);
            try (AbstractTensor ignored = model.batchForward(prompt, 0, cachedSession)) {
                cachedTrace.clear();
            }
            try (AbstractTensor ignored = model.forward(continuation, prompt.length, cachedSession)) {
                // capture cached decode
            }
            model.setLayerDebugHook(coldTrace::record);
            try (AbstractTensor ignored = model.batchForward(replayTokens, 0, coldSession)) {
                // capture cold replay
            }
        }

        compareTraces(layerSnapshots(cachedTrace.snapshots, 1), layerSnapshots(coldTrace.snapshots, 1));
    }

    @Test
    void nativeVersusPanamaLayer1CachedTrace() throws Exception {
        Path modelDir = Qwen3HfTextModelPortedTest.writeTinyCheckpoint(
                tempDir.resolve("qwen3-native-panama-layer1"),
                Qwen3HfTextModelPortedTest.tinyConfig(), 12_346);
        int[] prompt = new int[]{3, 4, 5, 6};
        int continuation = 7;
        TraceRecorder nativeTrace = new TraceRecorder("native");
        TraceRecorder panamaTrace = new TraceRecorder("panama");

        try (Qwen3Model nativeModel = Qwen3HfTextModelPortedTest.loadTinyQwen3Model(modelDir);
             Qwen3Model panamaModel = Qwen3HfTextModelPortedTest.loadTinyQwen3Model(modelDir);
             KvCacheSession nativeSession = nativeModel.newKvCacheSession();
             KvCacheSession panamaSession = panamaModel.newKvCacheSession()) {
            panamaModel.getLighter().putTensorOperations(TensorProviderKind.SIMD,
                    panamaModel.getLighter().tensorOperations().get(TensorProviderKind.PANAMA));
            TensorOps nativeOps = nativeModel.getLighter().tensorOperations().get(TensorProviderKind.SIMD);
            assertNotNull(nativeOps, "native Tensor2 provider must be present for native/Panama parity");
            nativeModel.setLayerDebugHook(nativeTrace::record);
            panamaModel.setLayerDebugHook(panamaTrace::record);
            try (AbstractTensor ignoredNative = nativeModel.batchForward(prompt, 0, nativeSession);
                 AbstractTensor ignoredPanama = panamaModel.batchForward(prompt, 0, panamaSession)) {
                writeProviderTracesIfRequested(nativeTrace, panamaTrace);
                compareCommonTraces(nativeTrace.snapshots, panamaTrace.snapshots, "native versus Panama prefill");
                nativeTrace.clear();
                panamaTrace.clear();
            }
            try (AbstractTensor ignoredNative = nativeModel.forward(continuation, prompt.length, nativeSession);
                 AbstractTensor ignoredPanama = panamaModel.forward(continuation, prompt.length, panamaSession)) {
                // capture cached decode traces
            }
        }

        writeProviderTracesIfRequested(nativeTrace, panamaTrace);
        compareCommonTraces(layerSnapshots(nativeTrace.snapshots, 1),
                layerSnapshots(panamaTrace.snapshots, 1), "native versus Panama");
    }

    @Test
    void actualLoadedOProjectionProvidersMatchCachedAndColdInputRows() throws Exception {
        Path modelDir = Qwen3HfTextModelPortedTest.writeTinyCheckpoint(
                tempDir.resolve("qwen3-actual-o-projection-replay"),
                Qwen3HfTextModelPortedTest.tinyConfig(), 12_346);
        int[] prompt = new int[]{3, 4, 5, 6};
        int continuation = 7;
        int[] replayTokens = Arrays.copyOf(prompt, prompt.length + 1);
        replayTokens[prompt.length] = continuation;
        Map<String, float[]> cached = new LinkedHashMap<>();
        Map<String, float[]> cold = new LinkedHashMap<>();

        try (Qwen3Model model = Qwen3HfTextModelPortedTest.loadTinyQwen3Model(modelDir);
             KvCacheSession cachedSession = model.newKvCacheSession();
             KvCacheSession coldSession = model.newKvCacheSession()) {
            model.setLayerDebugHook(event -> {
                if (event.layerIndex() == 1 && "attention_output_projection_input".equals(event.stage())) {
                    cached.put("input", lastRowValues(event.hiddenStates()));
                }
            });
            try (AbstractTensor ignored = model.batchForward(prompt, 0, cachedSession);
                 AbstractTensor decode = model.forward(continuation, prompt.length, cachedSession)) {
                assertFinite(decode);
            }
            model.setLayerDebugHook(event -> {
                if (event.layerIndex() == 1 && "attention_output_projection_input".equals(event.stage())) {
                    cold.put("input", lastRowValues(event.hiddenStates()));
                }
            });
            try (AbstractTensor replay = model.batchForward(replayTokens, 0, coldSession)) {
                assertFinite(replay);
            }

            assertEquals(cold.keySet(), cached.keySet(), "projection input capture");
            float inputMaxAbs = 0.0f;
            for (int column = 0; column < cached.get("input").length; column++) {
                inputMaxAbs = Math.max(inputMaxAbs,
                        Math.abs(cold.get("input")[column] - cached.get("input")[column]));
            }
            System.out.println("actual o_proj input cached/cold maxAbs=" + inputMaxAbs);

            TensorRef outputWeights = actualOutputProjectionWeights(model, 1);
            Lighter source = model.getLighter();
            try (TensorRef replayF32 = source.allocate(DType.F32, TensorShape.of(1, cached.get("input").length))) {
                fillRow(replayF32, cached.get("input"));
                try (TensorRef replayI8 = source.reshape(replayF32, DType.I8);
                     TensorRef sequential = model.makeDenseTensorRef(1, 32);
                     TensorRef chunked = model.makeDenseTensorRef(1, 32)) {
                    for (int chunkStart = 0; chunkStart < 32; chunkStart += 8) {
                        source.dotProductRows(sequential, replayI8, outputWeights, 0, 32,
                                chunkStart, 8, chunkStart);
                    }
                    model.runChunks("diagnostic.o_proj", 0, 32, 32, java.util.Optional.empty(),
                            (chunkStart, chunkSize) -> source.dotProductRows(chunked, replayI8, outputWeights,
                                    0, 32, chunkStart, chunkSize, chunkStart));
                    for (int column = 0; column < 32; column++) {
                        assertEquals(sequential.get(0, column), chunked.get(0, column), 0.03f,
                                "actual provider sequential versus runChunks column=" + column);
                    }
                }
            }
            for (Map.Entry<TensorProviderKind, TensorOps> provider : source.tensorOperations().entrySet()) {
                if (provider.getKey() == TensorProviderKind.GPU) {
                    continue;
                }
                Lighter actual = new Lighter(new io.dropwizard.metrics5.MetricRegistry(),
                        Map.of(provider.getKey(), provider.getValue()));
                Lighter expected = new Lighter(new io.dropwizard.metrics5.MetricRegistry(),
                        Map.of(TensorProviderKind.NAIVE, source.tensorOperations().get(TensorProviderKind.NAIVE)));
                try (TensorRef cachedF32 = actual.allocate(DType.F32, TensorShape.of(1, cached.get("input").length));
                     TensorRef coldF32 = actual.allocate(DType.F32, TensorShape.of(1, cold.get("input").length))) {
                    fillRow(cachedF32, cached.get("input"));
                    fillRow(coldF32, cold.get("input"));
                    try (TensorRef cachedI8 = source.reshape(cachedF32, DType.I8);
                          TensorRef coldI8 = source.reshape(coldF32, DType.I8);
                          TensorRef actualCached = actual.allocate(DType.F32, TensorShape.of(1, 32));
                          TensorRef actualParallel = actual.allocate(DType.F32, TensorShape.of(1, 32));
                          TensorRef expectedCached = expected.allocate(DType.F32, TensorShape.of(1, 32))) {
                        for (int chunkStart = 0; chunkStart < 32; chunkStart++) {
                            actual.dotProductRows(actualCached, cachedI8, outputWeights, 0, 32,
                                    chunkStart, 1, chunkStart);
                             expected.dotProductRows(expectedCached, cachedI8, outputWeights, 0, 32,
                                     chunkStart, 1, chunkStart);
                         }
                         model.runChunks("diagnostic.provider.o_proj", 0, 32, 32, java.util.Optional.empty(),
                                 (chunkStart, chunkSize) -> actual.dotProductRows(actualParallel, cachedI8,
                                         outputWeights, 0, 32, chunkStart, chunkSize, chunkStart));
                         for (int column = 0; column < 32; column++) {
                             assertEquals(expectedCached.get(0, column), actualCached.get(0, column), 0.03f,
                                     provider.getKey() + " oracle column=" + column);
                             assertEquals(expectedCached.get(0, column), actualParallel.get(0, column), 0.03f,
                                     provider.getKey() + " parallel oracle column=" + column);
                         }
                    }
                }
            }
        }
    }

    private static boolean isAttentionStage(String stage) {
        return "attention_projection_input".equals(stage)
                || "query_projection".equals(stage)
                || "key_projection".equals(stage)
                || "value_projection".equals(stage)
                || "query_normalized".equals(stage)
                || "key_normalized".equals(stage)
                || "query_rope".equals(stage)
                || "key_rope".equals(stage)
                || "attention_value".equals(stage);
    }

    private static float[] lastRowValues(AbstractTensor tensor) {
        int row = tensor.shape().first() - 1;
        float[] values = new float[tensor.shape().last()];
        for (int column = 0; column < values.length; column++) {
            values[column] = tensor.get(row, column);
        }
        return values;
    }

    private static void fillRow(TensorRef target, float[] values) {
        for (int column = 0; column < values.length; column++) {
            target.set(values[column], 0, column);
        }
    }

    private static TensorRef actualOutputProjectionWeights(Qwen3Model model, int layerIndex) throws Exception {
        Field blocksField = AbstractModel.class.getDeclaredField("transformerBlocks2");
        blocksField.setAccessible(true);
        TransformerBlock2[] blocks = (TransformerBlock2[]) blocksField.get(model);
        Field attentionField = TransformerBlock2.class.getDeclaredField("attention");
        attentionField.setAccessible(true);
        KvCacheSelfAttention2 attention = (KvCacheSelfAttention2) attentionField.get(blocks[layerIndex]);
        Field weightsField = KvCacheSelfAttention2.class.getDeclaredField("outputProjectionWeights");
        weightsField.setAccessible(true);
        return (TensorRef) weightsField.get(attention);
    }

    private static void assertFinite(AbstractTensor tensor) {
        for (int row = 0; row < tensor.shape().first(); row++) {
            for (int column = 0; column < tensor.shape().last(); column++) {
                assertTrue(Float.isFinite(tensor.get(row, column)),
                        "non-finite value row=" + row + " column=" + column);
            }
        }
    }

    private static void writeTracesIfRequested(TraceRecorder cached, TraceRecorder cold) {
        String configuredDirectory = System.getProperty("qwen3.trace.dir");
        if (configuredDirectory == null || configuredDirectory.isBlank()) {
            return;
        }
        Path directory = Path.of(configuredDirectory);
        try {
            java.nio.file.Files.createDirectories(directory);
            cached.write(directory.resolve("cached.jsonl"));
            cold.write(directory.resolve("cold.jsonl"));
            System.out.println("wrote Qwen3 traces to " + directory);
        } catch (IOException e) {
            throw new RuntimeException("Unable to write Qwen3 tensor traces to " + directory, e);
        }
    }

    private static void writeProviderTracesIfRequested(TraceRecorder nativeTrace, TraceRecorder panamaTrace) {
        String configuredDirectory = System.getProperty("qwen3.trace.dir");
        if (configuredDirectory == null || configuredDirectory.isBlank()) {
            return;
        }
        Path directory = Path.of(configuredDirectory);
        try {
            java.nio.file.Files.createDirectories(directory);
            nativeTrace.write(directory.resolve("native.jsonl"));
            panamaTrace.write(directory.resolve("panama.jsonl"));
            System.out.println("wrote provider traces to " + directory);
        } catch (IOException e) {
            throw new RuntimeException("Unable to write provider traces to " + directory, e);
        }
    }

    private static void compareTraces(List<TensorSnapshot> cached, List<TensorSnapshot> cold) {
        Map<String, TensorSnapshot> cachedByKey = traceByKey(cached);
        Map<String, TensorSnapshot> coldByKey = traceByKey(cold);
        assertEquals(coldByKey.keySet(), cachedByKey.keySet(), "cached versus cold trace stages");
        int index = 0;
        for (String key : cachedByKey.keySet().stream().sorted().toList()) {
            TensorSnapshot left = cachedByKey.get(key);
            TensorSnapshot right = coldByKey.get(key);
            assertEquals(left.dtype(), right.dtype(), "trace dtype key=" + key);
            int columns = left.shape().length > 1 ? left.shape()[1] : left.shape()[0];
            assertEquals(columns, right.shape().length > 1 ? right.shape()[1] : right.shape()[0],
                    "trace width event=" + index);
            boolean compareColdLastRow = left.values().length != right.values().length;
            int limit = left.values().length;
            for (int valueIndex = 0; valueIndex < limit; valueIndex++) {
                int coldIndex = compareColdLastRow
                        ? (right.values().length - columns) + (valueIndex % columns)
                        : valueIndex;
                float difference = Math.abs(left.values()[valueIndex] - right.values()[coldIndex]);
                if (difference > 1.0e-4f) {
                    int row = columns == 0 ? 0 : valueIndex / columns;
                    int column = columns == 0 ? valueIndex : valueIndex % columns;
                    assertTrue(false, "first cached/cold trace difference path=" + left.path()
                            + " layer=" + left.layer() + " stage=" + left.stage()
                            + " row=" + row + " column=" + column
                            + " cached=" + left.values()[valueIndex]
                            + " cold=" + right.values()[coldIndex]
                            + " absDiff=" + difference);
                }
            }
            index++;
        }
    }

    private static void compareCommonTraces(List<TensorSnapshot> actual, List<TensorSnapshot> expected,
            String label) {
        Map<String, TensorSnapshot> actualByKey = traceByKey(actual);
        Map<String, TensorSnapshot> expectedByKey = traceByKey(expected);
        List<String> common = actualByKey.keySet().stream().filter(expectedByKey::containsKey).toList();
        assertTrue(!common.isEmpty(), label + " has no common trace stages");
        for (String key : common) {
            TensorSnapshot left = actualByKey.get(key);
            TensorSnapshot right = expectedByKey.get(key);
            assertEquals(left.dtype(), right.dtype(), label + " trace dtype key=" + key);
            int columns = left.shape().length > 1 ? left.shape()[1] : left.shape()[0];
            assertEquals(columns, right.shape().length > 1 ? right.shape()[1] : right.shape()[0],
                    label + " trace width key=" + key);
            boolean compareLastRow = left.values().length != right.values().length;
            for (int valueIndex = 0; valueIndex < left.values().length; valueIndex++) {
                int expectedIndex = compareLastRow
                        ? right.values().length - columns + valueIndex % columns
                        : valueIndex;
                float difference = Math.abs(left.values()[valueIndex] - right.values()[expectedIndex]);
                if (difference > 1.0e-3f) {
                    int row = valueIndex / columns;
                    int column = valueIndex % columns;
                    assertTrue(false, label + " first legacy/TensorRef difference stage=" + left.stage()
                            + " layer=" + left.layer() + " row=" + row + " column=" + column
                            + " tensorRef=" + left.values()[valueIndex]
                            + " legacy=" + right.values()[expectedIndex]
                            + " absDiff=" + difference);
                }
            }
        }
    }

    private static Map<String, TensorSnapshot> traceByKey(List<TensorSnapshot> snapshots) {
        Map<String, TensorSnapshot> byKey = new LinkedHashMap<>();
        for (TensorSnapshot snapshot : snapshots) {
            byKey.put(snapshot.layer() + ":" + snapshot.stage(), snapshot);
        }
        return byKey;
    }

    private static List<TensorSnapshot> layerSnapshots(List<TensorSnapshot> snapshots, int layer) {
        return snapshots.stream().filter(snapshot -> snapshot.layer() == layer).toList();
    }

    private static final class TraceRecorder {
        private final String path;
        private final List<TensorSnapshot> snapshots = new java.util.ArrayList<>();

        private TraceRecorder(String path) {
            this.path = path;
        }

        private void record(AbstractModel.LayerDebugEvent event) {
            snapshots.add(new TensorSnapshot(path, event.layerIndex(), event.stage(),
                    event.hiddenStates().dType().name(), event.hiddenStates().getStride(),
                    event.hiddenStates().shape().shapeArray(), values(event.hiddenStates())));
        }

        private void clear() {
            snapshots.clear();
        }

        private void write(Path output) throws IOException {
            try (BufferedWriter writer = java.nio.file.Files.newBufferedWriter(output)) {
                for (TensorSnapshot snapshot : snapshots) {
                    writer.write(JsonUtils.om.writeValueAsString(snapshot));
                    writer.newLine();
                }
            }
        }

        private static float[] values(AbstractTensor tensor) {
            float[] values = new float[(int) tensor.size()];
            int index = 0;
            for (int row = 0; row < tensor.shape().first(); row++) {
                for (int column = 0; column < tensor.shape().last(); column++) {
                    values[index++] = tensor.get(row, column);
                }
            }
            return values;
        }
    }

    private record TensorSnapshot(String path, int layer, String stage, String dtype, long stride, int[] shape,
            float[] values) {
    }
}
