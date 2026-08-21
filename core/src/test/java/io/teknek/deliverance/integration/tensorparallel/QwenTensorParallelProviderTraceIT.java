package io.teknek.deliverance.integration.tensorparallel;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.math.VectorMath;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.model.tensorparallel.InProcessTensorParallelCollectives;
import io.teknek.deliverance.model.tensorparallel.StaticTensorParallelContext;
import io.teknek.deliverance.model.tensorparallel.TensorParallelGenerationGroup;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.ReadOnlyTensor;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.impl.Q8ByteBufferTensor;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.MachineSpec;
import io.teknek.deliverance.tensor.operations.NativeSimdTensorOperations;
import io.teknek.deliverance.tensor.operations.PanamaTensorOperations;
import io.teknek.deliverance.tensor.operations.TensorOperations;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.fail;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

public class QwenTensorParallelProviderTraceIT {
    private static final int TENSOR_PARALLEL_SIZE = 4;
    private static final float ABS_TOLERANCE = 2.0e-2f;
    private static final float REL_TOLERANCE = 2.0e-2f;

    @Test
    @Tag("longtest")
    public void qwen06bTpTraceFindsFirstNativeProviderDivergence() {
        ModelFetcher fetcher = new ModelFetcher("edwardcapriolo", "Qwen3-0.6B-JQ4");
        TraceRun panama = runTrace(fetcher, ProviderMode.PANAMA);
        TraceRun nativeSimd = runTrace(fetcher, ProviderMode.DEFAULT_NATIVE);

        assumeTrue(nativeSimd.providerNames().stream().anyMatch(name -> name.toLowerCase().contains("native")),
                "native SIMD provider is not available; default provider fell back to " + nativeSimd.providerNames());
        compareTraces(panama, nativeSimd);
    }

    @Test
    @Tag("longtest")
    public void qwen06bTpLayer0MlpGateUpProjectionNativeReplayMatchesPanama() {
        ModelFetcher fetcher = new ModelFetcher("edwardcapriolo", "Qwen3-0.6B-JQ4");
        AtomicReference<AbstractTensor> capturedInput = new AtomicReference<>();
        InProcessTensorParallelCollectives.Group collectivesGroup =
                new InProcessTensorParallelCollectives.Group(Duration.ofSeconds(60));
        List<RankModel> rankModels = new ArrayList<>();
        try {
            for (int rank = 0; rank < TENSOR_PARALLEL_SIZE; rank++) {
                RankModel rankModel = buildRankModel(fetcher, ProviderMode.DEFAULT_NATIVE, rank, collectivesGroup,
                        event -> {
                            if (event.tensorParallelContext().rank() == 0 && event.layerIndex() == 0
                                    && "mlp_projection_input".equals(event.stage())) {
                                capturedInput.compareAndSet(null, copyTensorForReplay(event.hiddenStates()));
                            }
                        });
                rankModels.add(rankModel);
            }
            List<AbstractModel> models = rankModels.stream().map(RankModel::model).toList();
            assumeTrue(models.stream().map(AbstractModel::getTensorProviderName)
                            .anyMatch(name -> name.toLowerCase().contains("native")),
                    "native SIMD provider is not available; default provider fell back to "
                            + models.stream().map(AbstractModel::getTensorProviderName).toList());
            int[] promptTokens = models.getFirst().constructPromptTokensForRuntime("hi");
            try (TensorParallelGenerationGroup group = new TensorParallelGenerationGroup(models)) {
                List<AbstractTensor> outputs = group.batchForwardAllRanks(promptTokens, 0);
                outputs.forEach(AbstractTensor::close);
            }

            AbstractTensor input = capturedInput.get();
            if (input == null) {
                fail("did not capture rank=0 layer=0 mlp_projection_input");
            }
            MlpWeights weights = extractLayer0MlpWeights(models.getFirst());
            replayGateUpProjection(input, weights);
        } finally {
            AbstractTensor input = capturedInput.get();
            if (input != null) {
                input.close();
            }
            rankModels.forEach(RankModel::close);
        }
    }

    private static TraceRun runTrace(ModelFetcher fetcher, ProviderMode providerMode) {
        TraceRecorder recorder = new TraceRecorder(providerMode.name());
        InProcessTensorParallelCollectives.Group collectivesGroup =
                new InProcessTensorParallelCollectives.Group(Duration.ofSeconds(60));
        List<RankModel> rankModels = new ArrayList<>();
        try {
            for (int rank = 0; rank < TENSOR_PARALLEL_SIZE; rank++) {
                rankModels.add(buildRankModel(fetcher, providerMode, rank, collectivesGroup, recorder));
            }
            List<AbstractModel> models = rankModels.stream().map(RankModel::model).toList();
            int[] promptTokens = models.getFirst().constructPromptTokensForRuntime("hi");
            try (TensorParallelGenerationGroup group = new TensorParallelGenerationGroup(models)) {
                List<AbstractTensor> outputs = group.batchForwardAllRanks(promptTokens, 0);
                outputs.forEach(AbstractTensor::close);
            }
            return new TraceRun(providerMode.name(),
                    models.stream().map(AbstractModel::getTensorProviderName).toList(), recorder.snapshotsByRank());
        } finally {
            rankModels.forEach(RankModel::close);
        }
    }

    private static RankModel buildRankModel(ModelFetcher fetcher, ProviderMode providerMode, int rank,
            InProcessTensorParallelCollectives.Group collectivesGroup, TraceRecorder recorder) {
        return buildRankModel(fetcher, providerMode, rank, collectivesGroup, recorder::record);
    }

    private static RankModel buildRankModel(ModelFetcher fetcher, ProviderMode providerMode, int rank,
            InProcessTensorParallelCollectives.Group collectivesGroup,
            java.util.function.Consumer<AbstractModel.LayerDebugEvent> debugHook) {
        MetricRegistry metrics = new MetricRegistry();
        WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores());
        TensorAllocator allocator = new ArrayQueueTensorAllocator(metrics);
        StaticTensorParallelContext context = new StaticTensorParallelContext(rank, TENSOR_PARALLEL_SIZE);
        AutoModelForCausaLm.Builder builder = AutoModelForCausaLm.newBuilder(fetcher)
                .withMetricRegistry(metrics)
                .withWrappedForkJoinPool(pool)
                .withTensorAllocator(allocator)
                .withTensorParallelContext(context)
                .withTensorParallelCollectives(new InProcessTensorParallelCollectives(context, collectivesGroup));
        if (providerMode == ProviderMode.PANAMA) {
            builder.withTensorProvider(new ConfigurableTensorProvider(
                    new PanamaTensorOperations(MachineSpec.VECTOR_TYPE, allocator, pool)));
        }
        AbstractModel model = builder.buildLocalTransformerModel();
        model.setLayerDebugHook(debugHook);
        return new RankModel(model, pool);
    }

    private static void replayGateUpProjection(AbstractTensor input, MlpWeights weights) {
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores())) {
            TensorAllocator allocator = new ArrayQueueTensorAllocator(new MetricRegistry());
            TensorOperations panama = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE, allocator, pool);
            TensorOperations nativeSimd = new NativeSimdTensorOperations(panama);
            int batchSize = input.shape().first();
            int embeddingLength = input.shape().last();
            int localHiddenLength = weights.gateProjectionWeights().shape().first();
            try (FloatBufferTensor expectedGate = new FloatBufferTensor(batchSize, localHiddenLength);
                 FloatBufferTensor expectedUp = new FloatBufferTensor(batchSize, localHiddenLength);
                 FloatBufferTensor actualGate = new FloatBufferTensor(batchSize, localHiddenLength);
                 FloatBufferTensor actualUp = new FloatBufferTensor(batchSize, localHiddenLength)) {
                AbstractTensor[] weightsArray = {weights.gateProjectionWeights(), weights.upProjectionWeights()};
                panama.registerModelTensor(weights.gateProjectionWeights());
                panama.registerModelTensor(weights.upProjectionWeights());
                nativeSimd.registerModelTensor(weights.gateProjectionWeights());
                nativeSimd.registerModelTensor(weights.upProjectionWeights());
                VectorMath.pchunk(0, localHiddenLength,
                        (chunkStart, chunkSize) -> panama.dotProductBatchChunk(
                                new AbstractTensor[]{expectedGate, expectedUp}, input, weightsArray,
                                0, embeddingLength, chunkStart, chunkSize),
                        panama.parallelSplitSize(), pool);
                VectorMath.pchunk(0, localHiddenLength,
                        (chunkStart, chunkSize) -> nativeSimd.dotProductBatchChunk(
                                new AbstractTensor[]{actualGate, actualUp}, input, weightsArray,
                                0, embeddingLength, chunkStart, chunkSize),
                        nativeSimd.parallelSplitSize(), pool);
                assertReplayTensorClose("gate", expectedGate, actualGate);
                assertReplayTensorClose("up", expectedUp, actualUp);
            }
        }
    }

    private static void assertReplayTensorClose(String label, AbstractTensor expected, AbstractTensor actual) {
        Diff diff = firstDiff(new TensorSnapshot(0, 0, label, expected.shape().shapeArray(), values(expected)),
                new TensorSnapshot(0, 0, label, actual.shape().shapeArray(), values(actual)));
        if (diff != null) {
            TensorSnapshot snapshot = new TensorSnapshot(0, 0, label, expected.shape().shapeArray(), values(expected));
            fail("focused native replay divergence stage=mlp_" + label + "_projection"
                    + " shape=" + Arrays.toString(expected.shape().shapeArray())
                    + " maxAbsDiff=" + diff.maxAbsDiff()
                    + " maxAbsDiffIndex=" + diff.maxAbsDiffIndex()
                    + " maxAbsDiffPosition=" + position(snapshot, diff.maxAbsDiffIndex())
                    + " firstDiffIndex=" + diff.firstDiffIndex()
                    + " firstDiffPosition=" + position(snapshot, diff.firstDiffIndex())
                    + " expected=" + diff.expected()
                    + " actual=" + diff.actual()
                    + " absDiff=" + diff.absDiff());
        }
    }

    private static MlpWeights extractLayer0MlpWeights(AbstractModel model) {
        Object[] transformerBlocks = (Object[]) getField(AbstractModel.class, model, "transformerBlocks");
        Object ffBlock = getField(transformerBlocks[0].getClass(), transformerBlocks[0], "ffBlock");
        return new MlpWeights((AbstractTensor) getField(ffBlock.getClass(), ffBlock, "fullyConnectedWeights"),
                (AbstractTensor) getField(ffBlock.getClass(), ffBlock, "upProjectionWeights"));
    }

    private static Object getField(Class<?> owner, Object target, String name) {
        try {
            Field field = owner.getDeclaredField(name);
            field.setAccessible(true);
            return field.get(target);
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException("Unable to read field " + owner.getName() + "." + name, e);
        }
    }

    private static AbstractTensor copyTensorForReplay(AbstractTensor source) {
        AbstractTensor unwrapped = unwrapReadOnly(source);
        if (unwrapped instanceof Q8ByteBufferTensor q8) {
            Q8ByteBufferTensor copy = new Q8ByteBufferTensor(q8.shape());
            copy.copyFrom(q8, 0, 0, (int) q8.size());
            copy.getBlockF().copyFrom(q8.getBlockF(), 0, 0, (int) q8.getBlockF().size());
            return copy;
        }
        if (unwrapped instanceof FloatBufferTensor) {
            return new FloatBufferTensor(unwrapped);
        }
        throw new IllegalArgumentException("Unsupported replay tensor type: " + unwrapped.getClass().getName()
                + " dtype=" + unwrapped.dType());
    }

    private static AbstractTensor unwrapReadOnly(AbstractTensor tensor) {
        AbstractTensor current = tensor;
        while (current instanceof ReadOnlyTensor readOnly) {
            current = readOnly.delegate();
        }
        return current;
    }

    private static void compareTraces(TraceRun expected, TraceRun actual) {
        for (int rank = 0; rank < TENSOR_PARALLEL_SIZE; rank++) {
            List<TensorSnapshot> expectedRank = expected.snapshotsByRank().getOrDefault(rank, List.of());
            List<TensorSnapshot> actualRank = actual.snapshotsByRank().getOrDefault(rank, List.of());
            if (expectedRank.size() != actualRank.size()) {
                fail("trace event count mismatch rank=" + rank
                        + " expectedProvider=" + expected.providerMode()
                        + " actualProvider=" + actual.providerMode()
                        + " expectedEvents=" + expectedRank.size()
                        + " actualEvents=" + actualRank.size());
            }
            for (int i = 0; i < expectedRank.size(); i++) {
                TensorSnapshot left = expectedRank.get(i);
                TensorSnapshot right = actualRank.get(i);
                compareSnapshotMetadata(rank, i, left, right, expected, actual);
                Diff diff = firstDiff(left, right);
                if (diff != null) {
                    fail("first provider trace divergence rank=" + rank
                            + " eventIndex=" + i
                            + " layer=" + left.layerIndex()
                            + " stage=" + left.stage()
                            + " shape=" + Arrays.toString(left.shape())
                            + " expectedProvider=" + expected.providerMode()
                            + " actualProvider=" + actual.providerMode()
                            + " maxAbsDiff=" + diff.maxAbsDiff()
                            + " maxAbsDiffIndex=" + diff.maxAbsDiffIndex()
                            + " maxAbsDiffPosition=" + position(left, diff.maxAbsDiffIndex())
                            + " firstDiffIndex=" + diff.firstDiffIndex()
                            + " firstDiffPosition=" + position(left, diff.firstDiffIndex())
                            + " expected=" + diff.expected()
                            + " actual=" + diff.actual()
                            + " absDiff=" + diff.absDiff());
                }
            }
        }
    }

    private static void compareSnapshotMetadata(int rank, int eventIndex, TensorSnapshot expected,
            TensorSnapshot actual, TraceRun expectedRun, TraceRun actualRun) {
        if (expected.layerIndex() != actual.layerIndex() || !expected.stage().equals(actual.stage())
                || !Arrays.equals(expected.shape(), actual.shape())) {
            fail("trace metadata mismatch rank=" + rank
                    + " eventIndex=" + eventIndex
                    + " expectedProvider=" + expectedRun.providerMode()
                    + " actualProvider=" + actualRun.providerMode()
                    + " expectedLayer=" + expected.layerIndex()
                    + " actualLayer=" + actual.layerIndex()
                    + " expectedStage=" + expected.stage()
                    + " actualStage=" + actual.stage()
                    + " expectedShape=" + Arrays.toString(expected.shape())
                    + " actualShape=" + Arrays.toString(actual.shape()));
        }
    }

    private static Diff firstDiff(TensorSnapshot expected, TensorSnapshot actual) {
        int maxAbsDiffIndex = -1;
        float maxAbsDiff = 0.0f;
        int firstDiffIndex = -1;
        float firstExpected = 0.0f;
        float firstActual = 0.0f;
        float firstAbsDiff = 0.0f;
        for (int i = 0; i < expected.values().length; i++) {
            float left = expected.values()[i];
            float right = actual.values()[i];
            float absDiff = Math.abs(left - right);
            if (absDiff > maxAbsDiff) {
                maxAbsDiff = absDiff;
                maxAbsDiffIndex = i;
            }
            float tolerance = ABS_TOLERANCE + REL_TOLERANCE * Math.max(Math.abs(left), Math.abs(right));
            if (firstDiffIndex < 0 && absDiff > tolerance) {
                firstDiffIndex = i;
                firstExpected = left;
                firstActual = right;
                firstAbsDiff = absDiff;
            }
        }
        if (firstDiffIndex < 0) {
            return null;
        }
        return new Diff(firstDiffIndex, firstExpected, firstActual, firstAbsDiff, maxAbsDiffIndex, maxAbsDiff);
    }

    private static String position(TensorSnapshot snapshot, int index) {
        if (snapshot.shape().length == 2) {
            int columns = snapshot.shape()[1];
            return "row=" + (index / columns) + ",column=" + (index % columns);
        }
        return "linear=" + index;
    }

    private static TensorSnapshot snapshot(AbstractModel.LayerDebugEvent event) {
        AbstractTensor tensor = event.hiddenStates();
        int[] shape = tensor.shape().shapeArray();
        return new TensorSnapshot(event.tensorParallelContext().rank(), event.layerIndex(), event.stage(), shape,
                values(tensor));
    }

    private static float[] values(AbstractTensor tensor) {
        float[] values = new float[(int) tensor.size()];
        if (tensor.dims() == 2) {
            int index = 0;
            for (int row = 0; row < tensor.shape().first(); row++) {
                for (int column = 0; column < tensor.shape().last(); column++) {
                    values[index++] = tensor.get(row, column);
                }
            }
        } else {
            int[] cursor = new int[tensor.dims()];
            for (int i = 0; i < values.length; i++) {
                values[i] = tensor.get(cursor);
                tensor.iterate(cursor);
            }
        }
        return values;
    }

    private enum ProviderMode {
        PANAMA,
        DEFAULT_NATIVE
    }

    private static final class TraceRecorder {
        private final Map<Integer, List<TensorSnapshot>> snapshotsByRank = new ConcurrentHashMap<>();

        private TraceRecorder(String ignoredProviderMode) {
        }

        private void record(AbstractModel.LayerDebugEvent event) {
            int rank = event.tensorParallelContext().rank();
            snapshotsByRank.computeIfAbsent(rank, ignored -> new ArrayList<>()).add(snapshot(event));
        }

        private Map<Integer, List<TensorSnapshot>> snapshotsByRank() {
            return snapshotsByRank.entrySet().stream()
                    .collect(java.util.stream.Collectors.toUnmodifiableMap(Map.Entry::getKey,
                            entry -> List.copyOf(entry.getValue())));
        }
    }

    private record TraceRun(String providerMode, List<String> providerNames,
            Map<Integer, List<TensorSnapshot>> snapshotsByRank) {
    }

    private record TensorSnapshot(int rank, int layerIndex, String stage, int[] shape, float[] values) {
    }

    private record MlpWeights(AbstractTensor gateProjectionWeights, AbstractTensor upProjectionWeights) {
    }

    private record Diff(int firstDiffIndex, float expected, float actual, float absDiff,
            int maxAbsDiffIndex, float maxAbsDiff) {
    }

    private record RankModel(AbstractModel model, WrappedForkJoinPool pool) implements AutoCloseable {
        @Override
        public void close() {
            model.close();
            pool.close();
        }
    }
}
