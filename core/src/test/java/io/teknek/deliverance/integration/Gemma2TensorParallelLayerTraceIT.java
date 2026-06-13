package io.teknek.deliverance.integration;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.CausalSelfAttention;
import io.teknek.deliverance.generator.LayerNorm;
import io.teknek.deliverance.generator.TransformerBlock;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.model.tensorparallel.InProcessTensorParallelCollectives;
import io.teknek.deliverance.model.tensorparallel.StaticTensorParallelContext;
import io.teknek.deliverance.model.tensorparallel.TensorParallelGenerationGroup;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.operations.MachineSpec;
import io.teknek.deliverance.tensor.operations.NativeSimdTensorOperations;
import io.teknek.deliverance.tensor.operations.PanamaTensorOperations;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.NaiveTensorOperations;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.lang.reflect.Field;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.BiFunction;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.fail;

public class Gemma2TensorParallelLayerTraceIT {
    private static final int TP_SIZE = 4;

    @Test
    public void traceSingleVsTensorParallelPrefillLayerOutputs() {
        traceSingleVsTensorParallelPrefillLayerOutputs(TensorProviderMode.PANAMA);
    }

    @Disabled("takes 30 seconds to run")
    public void traceSingleVsTensorParallelPrefillLayerOutputsWithNaiveTensorOperations() {
        traceSingleVsTensorParallelPrefillLayerOutputs(TensorProviderMode.NAIVE);
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("queryProjectionProviders")
    public void rankZeroLayerZeroQueryProjectionProviderBehavior(String providerName,
            BiFunction<ArrayQueueTensorAllocator, WrappedForkJoinPool, ConfigurableTensorProvider> providerFactory) {
        ModelFetcher fetcher = new ModelFetcher("tjake", "gemma-2-2b-it-JQ4");
        MetricRegistry metrics = new MetricRegistry();
        ArrayQueueTensorAllocator allocator = new ArrayQueueTensorAllocator(metrics);
        WrappedForkJoinPool pool = new WrappedForkJoinPool(new ForkJoinPool(Math.min(4,
                Runtime.getRuntime().availableProcessors())));

        try (pool;
             AbstractModel rankModel = AutoModelForCausaLm.newBuilder(fetcher)
                     .withMetricRegistry(metrics)
                      .withTensorAllocator(allocator)
                      .withWrappedForkJoinPool(pool)
                      .withWorkingQuantType(DType.F32)
                      .withTensorProvider(providerFactory.apply(allocator, pool))
                      .withTensorParallelContext(new StaticTensorParallelContext(0, TP_SIZE))
                     .withTensorParallelCollectives(new InProcessTensorParallelCollectives(
                             new StaticTensorParallelContext(0, TP_SIZE),
                             new InProcessTensorParallelCollectives.Group(Duration.ofSeconds(30))))
                     .buildLocalTransformerModel()) {
            String renderedPrompt = rankModel.promptSupport().get().builder()
                    .addUserMessage("What is tensor parallelism?")
                    .build()
                    .getPrompt();
            int[] promptTokens = constructPromptTokensLikeGenerate(rankModel, renderedPrompt);
            AtomicReference<AbstractTensor> modelInput = new AtomicReference<>();
            AtomicReference<AbstractTensor> queryProjection = new AtomicReference<>();
            rankModel.setLayerDebugHook(event -> {
                if (event.layerIndex() == -1 && event.stage().equals("input")) {
                    closeAndSet(modelInput, new FloatBufferTensor(event.hiddenStates()));
                }
                if (event.layerIndex() == 0 && event.stage().equals("query_projection")) {
                    closeAndSet(queryProjection, new FloatBufferTensor(event.hiddenStates()));
                    throw new StopAfterQueryProjection();
                }
            });

            try {
                rankModel.batchForward(promptTokens, 0).close();
                fail("expected query projection hook to stop the forward pass");
            } catch (StopAfterQueryProjection expected) {
                // debug hook captured the tensors before any tensor-parallel collective is reached
            } finally {
                rankModel.clearLayerDebugHook();
            }

            assertNotNull(modelInput.get(), "captured layer input");
            assertNotNull(queryProjection.get(), "captured query projection");
            try (AbstractTensor input = modelInput.get();
              AbstractTensor actual = queryProjection.get();
                  AbstractTensor direct = directLayerZeroQueryProjection(rankModel, input)) {
                assertTensorEquals(direct, actual, 0.001f);
            }
        }
    }

    private static Stream<Arguments> queryProjectionProviders() {
        return Stream.of(
                Arguments.of("panama", (BiFunction<ArrayQueueTensorAllocator, WrappedForkJoinPool, ConfigurableTensorProvider>)
                        (allocator, pool) -> new ConfigurableTensorProvider(
                                new PanamaTensorOperations(MachineSpec.VECTOR_TYPE, allocator, pool))),
                Arguments.of("native", (BiFunction<ArrayQueueTensorAllocator, WrappedForkJoinPool, ConfigurableTensorProvider>)
                        (allocator, pool) -> new ConfigurableTensorProvider(
                                new NativeSimdTensorOperations(new PanamaTensorOperations(MachineSpec.VECTOR_TYPE,
                                        allocator, pool))))
        );
    }

    private static void traceSingleVsTensorParallelPrefillLayerOutputs(TensorProviderMode tensorProviderMode) {
        ModelFetcher fetcher = new ModelFetcher("tjake", "gemma-2-2b-it-JQ4");
        MetricRegistry metrics = new MetricRegistry();
        ArrayQueueTensorAllocator allocator = new ArrayQueueTensorAllocator(metrics);
        WrappedForkJoinPool pool = new WrappedForkJoinPool(new ForkJoinPool(Math.min(4,
                Runtime.getRuntime().availableProcessors())));

        AutoModelForCausaLm.Builder singleBuilder = AutoModelForCausaLm.newBuilder(fetcher)
                .withMetricRegistry(metrics)
                .withTensorAllocator(allocator)
                .withWrappedForkJoinPool(pool)
                .withWorkingQuantType(DType.F32)
                .withTensorProvider(tensorProvider(tensorProviderMode, allocator, pool));
        try (pool;
             AbstractModel single = singleBuilder.buildLocalTransformerModel()) {
            String renderedPrompt = single.promptSupport().get().builder()
                    .addUserMessage("What is tensor parallelism?")
                    .build()
                    .getPrompt();
            int[] promptTokens = constructPromptTokensLikeGenerate(single, renderedPrompt);

            Map<String, RowSnapshot> singleLayers = new ConcurrentHashMap<>();
            single.setLayerDebugHook(event -> singleLayers.put(key(event.layerIndex(), event.stage()),
                    RowSnapshot.capture(event.hiddenStates())));
            try (AbstractTensor ignored = single.batchForward(promptTokens, 0)) {
                // layer hook captures summaries
            } finally {
                single.clearLayerDebugHook();
            }

            InProcessTensorParallelCollectives.Group collectiveGroup = new InProcessTensorParallelCollectives.Group(
                    Duration.ofSeconds(30));
            List<AbstractModel> rankModels = new ArrayList<>();
            Map<Integer, Map<String, RowSnapshot>> tpLayersByRank = new ConcurrentHashMap<>();
            for (int rank = 0; rank < TP_SIZE; rank++) {
                int rankId = rank;
                AutoModelForCausaLm.Builder rankBuilder = AutoModelForCausaLm.newBuilder(fetcher)
                        .withMetricRegistry(metrics)
                        .withTensorAllocator(allocator)
                        .withWrappedForkJoinPool(pool)
                        .withWorkingQuantType(DType.F32)
                        .withTensorProvider(tensorProvider(tensorProviderMode, allocator, pool))
                        .withTensorParallelContext(new StaticTensorParallelContext(rank, TP_SIZE))
                        .withTensorParallelCollectives(new InProcessTensorParallelCollectives(
                                new StaticTensorParallelContext(rank, TP_SIZE), collectiveGroup));
                AbstractModel rankModel = rankBuilder.buildLocalTransformerModel();
                rankModel.setLayerDebugHook(event -> tpLayersByRank
                        .computeIfAbsent(rankId, ignored -> new ConcurrentHashMap<>())
                        .put(key(event.layerIndex(), event.stage()), RowSnapshot.capture(event.hiddenStates())));
                rankModels.add(rankModel);
            }

            try (TensorParallelGenerationGroup group = new TensorParallelGenerationGroup(rankModels);
                 AbstractTensor ignored = group.batchForward(promptTokens, 0)) {
                compareLayerSnapshots(singleLayers, tpLayersByRank);
            }
        }
    }

    private static void compareLayerSnapshots(Map<String, RowSnapshot> singleLayers,
            Map<Integer, Map<String, RowSnapshot>> tpLayersByRank) {
        StringBuilder trace = new StringBuilder();
        List<String> keys = singleLayers.keySet().stream().sorted(Comparator.comparing(Gemma2TensorParallelLayerTraceIT::sortKey)).toList();
        for (String key : keys) {
            RowSnapshot single = singleLayers.get(key);
            RowSnapshot tp = requiresRankConcat(key)
                    ? combineRankSnapshots(tpLayersByRank, key)
                    : tpLayersByRank.getOrDefault(0, Map.of()).get(key);
            if (single == null || tp == null) {
                fail("missing layer snapshot key=" + key + " single=" + single + " tp=" + tp);
            }
            float diff = single.maxAbsDiff(tp);
            trace.append("key=").append(key)
                    .append(" diff=").append(diff)
                    .append(" single=").append(single.summary())
                    .append(" tp=").append(tp.summary())
                    .append('\n');
            if (diff > 0.5f) {
                System.out.println(trace);
                fail("first divergent key=" + key + " maxAbsDiff=" + diff
                        + " mismatch=" + single.mismatchSummary(tp)
                        + "\n" + trace);
            }
        }
        System.out.println(trace);
    }

    private static boolean requiresRankConcat(String key) {
        return key.endsWith(":query_projection")
                || key.endsWith(":key_projection")
                || key.endsWith(":value_projection")
                || key.endsWith(":attention_value");
    }

    private static RowSnapshot combineRankSnapshots(Map<Integer, Map<String, RowSnapshot>> tpLayersByRank, String key) {
        List<RowSnapshot> parts = new ArrayList<>();
        for (int rank = 0; rank < TP_SIZE; rank++) {
            RowSnapshot snapshot = tpLayersByRank.getOrDefault(rank, Map.of()).get(key);
            if (snapshot == null) {
                return null;
            }
            parts.add(snapshot);
        }
        int length = parts.stream().mapToInt(part -> part.row.length).sum();
        float[] row = new float[length];
        int offset = 0;
        for (RowSnapshot part : parts) {
            System.arraycopy(part.row, 0, row, offset, part.row.length);
            offset += part.row.length;
        }
        return RowSnapshot.capture(row);
    }

    private static String key(int layer, String stage) {
        return layer + ":" + stage;
    }

    private static String sortKey(String key) {
        String[] parts = key.split(":", 2);
        int layer = Integer.parseInt(parts[0]);
            int stageOrder = switch (parts[1]) {
            case "input" -> 0;
            case "query_projection" -> 1;
            case "key_projection" -> 2;
            case "value_projection" -> 3;
            case "attention_value" -> 4;
            case "attention_output" -> 5;
            case "post_attention_residual" -> 6;
            case "post_ff_residual" -> 7;
            case "layer_output" -> 8;
            default -> 9;
        };
        return String.format("%04d:%02d:%s", layer + 1, stageOrder, key);
    }

    private static int[] constructPromptTokensLikeGenerate(AbstractModel model, String renderedPrompt) {
        long[] encoded = model.encodeForRuntime(renderedPrompt);
        if (encoded.length > 0 && encoded[0] == model.getConfig().bosToken) {
            encoded = Arrays.copyOfRange(encoded, 1, encoded.length);
        }
        int[] promptTokens = new int[encoded.length + 1];
        promptTokens[0] = model.getConfig().bosToken;
        for (int i = 0; i < encoded.length; i++) {
            promptTokens[i + 1] = Math.toIntExact(encoded[i]);
        }
        return promptTokens;
    }

    private static AbstractTensor directLayerZeroQueryProjection(AbstractModel model, AbstractTensor modelInput) {
        TransformerBlock block = transformerBlocks(model)[0];
        Optional<LayerNorm> preAttentionNorm = field(block, TransformerBlock.class, "preAttentionNorm");
        CausalSelfAttention attention = field(block, TransformerBlock.class, "attention");
        AbstractTensor queryWeights = field(attention, CausalSelfAttention.class, "queryAttnWeights");

        AbstractTensor normalized = preAttentionNorm.map(norm -> norm.forward(modelInput)).orElse(modelInput);
        try (AbstractTensor qInput = model.maybeQuantize(normalized)) {
            AbstractTensor result = model.getTensorAllocator().get(DType.F32,
                    TensorShape.of(modelInput.shape().first(), model.getLocalAttentionLength()));
            new NaiveTensorOperations().dotProductChunk(result, qInput, queryWeights, 0, model.getConfig().embeddingLength,
                    0, model.getLocalAttentionLength());
            return result;
        } finally {
            if (normalized != modelInput) {
                normalized.close();
            }
        }
    }

    private static TransformerBlock[] transformerBlocks(AbstractModel model) {
        return field(model, AbstractModel.class, "transformerBlocks");
    }

    @SuppressWarnings("unchecked")
    private static <T> T field(Object target, Class<?> owner, String name) {
        try {
            Field field = owner.getDeclaredField(name);
            field.setAccessible(true);
            return (T) field.get(target);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException("Unable to read field " + owner.getSimpleName() + "." + name, e);
        }
    }

    private static void closeAndSet(AtomicReference<AbstractTensor> reference, AbstractTensor next) {
        AbstractTensor previous = reference.getAndSet(next);
        if (previous != null) {
            previous.close();
        }
    }

    private static void assertTensorEquals(AbstractTensor expected, AbstractTensor actual, float tolerance) {
        assertEquals(expected.shape().first(), actual.shape().first(), "row count");
        assertEquals(expected.shape().last(), actual.shape().last(), "column count");
        for (int row = 0; row < expected.shape().first(); row++) {
            for (int col = 0; col < expected.shape().last(); col++) {
                assertEquals(expected.get(row, col), actual.get(row, col), tolerance,
                        "row=" + row + " col=" + col + " expected=" + expected.get(row, col)
                                + " actual=" + actual.get(row, col));
            }
        }
    }

    private static final class StopAfterQueryProjection extends RuntimeException {
    }

    private enum TensorProviderMode {
        PANAMA,
        NAIVE
    }

    private static ConfigurableTensorProvider tensorProvider(TensorProviderMode mode, ArrayQueueTensorAllocator allocator,
            WrappedForkJoinPool pool) {
        return switch (mode) {
            case PANAMA -> new ConfigurableTensorProvider(new PanamaTensorOperations(MachineSpec.VECTOR_TYPE, allocator, pool));
            case NAIVE -> new ConfigurableTensorProvider(new NaiveTensorOperations());
        };
    }

    private record RowSnapshot(float[] row, float min, float max, float mean, float l2) {
        private static RowSnapshot capture(AbstractTensor tensor) {
            int rowIndex = tensor.shape().first() - 1;
            int width = tensor.shape().last();
            float[] row = new float[width];
            float min = Float.POSITIVE_INFINITY;
            float max = Float.NEGATIVE_INFINITY;
            double sum = 0.0;
            double sumSquares = 0.0;
            for (int col = 0; col < width; col++) {
                float value = tensor.get(rowIndex, col);
                row[col] = value;
                min = Math.min(min, value);
                max = Math.max(max, value);
                sum += value;
                sumSquares += (double) value * value;
            }
            return new RowSnapshot(row, min, max, (float) (sum / width), (float) Math.sqrt(sumSquares));
        }

        private static RowSnapshot capture(float[] row) {
            float min = Float.POSITIVE_INFINITY;
            float max = Float.NEGATIVE_INFINITY;
            double sum = 0.0;
            double sumSquares = 0.0;
            for (float value : row) {
                min = Math.min(min, value);
                max = Math.max(max, value);
                sum += value;
                sumSquares += (double) value * value;
            }
            return new RowSnapshot(row, min, max, (float) (sum / row.length), (float) Math.sqrt(sumSquares));
        }

        private float maxAbsDiff(RowSnapshot other) {
            float max = 0.0f;
            for (int i = 0; i < row.length; i++) {
                max = Math.max(max, Math.abs(row[i] - other.row[i]));
            }
            return max;
        }

        private String mismatchSummary(RowSnapshot other) {
            int first = -1;
            int maxIndex = -1;
            float maxDiff = 0.0f;
            for (int i = 0; i < row.length; i++) {
                float diff = Math.abs(row[i] - other.row[i]);
                if (diff != 0.0f && first == -1) {
                    first = i;
                }
                if (diff > maxDiff) {
                    maxDiff = diff;
                    maxIndex = i;
                }
            }
            return "firstIndex=" + first
                    + " firstSingle=" + (first >= 0 ? row[first] : Float.NaN)
                    + " firstTp=" + (first >= 0 ? other.row[first] : Float.NaN)
                    + " maxIndex=" + maxIndex
                    + " maxSingle=" + (maxIndex >= 0 ? row[maxIndex] : Float.NaN)
                    + " maxTp=" + (maxIndex >= 0 ? other.row[maxIndex] : Float.NaN)
                    + " maxDiff=" + maxDiff;
        }

        private String summary() {
            return "min=" + min + " max=" + max + " mean=" + mean + " l2=" + l2
                    + " first8=" + Arrays.toString(Arrays.copyOf(row, Math.min(8, row.length)));
        }
    }
}
