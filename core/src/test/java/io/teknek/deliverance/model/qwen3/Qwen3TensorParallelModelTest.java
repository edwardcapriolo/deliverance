package io.teknek.deliverance.model.qwen3;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.CausalSelfAttention;
import io.teknek.deliverance.generator.TransformerBlock;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.tensorparallel.InProcessTensorParallelCollectives;
import io.teknek.deliverance.model.tensorparallel.StaticTensorParallelContext;
import io.teknek.deliverance.model.tensorparallel.TensorParallelGenerationGroup;
import io.teknek.deliverance.safetensors.DefaultWeightLoader;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.TensorDisplayUtil;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.TensorTestSupport;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.MachineSpec;
import io.teknek.deliverance.tensor.operations.NaiveTensorOperations;
import io.teknek.deliverance.tensor.operations.NativeSimdTensorOperations;
import io.teknek.deliverance.tensor.operations.PanamaTensorOperations;
import io.teknek.deliverance.tensor.operations.TensorOperations;
import io.teknek.deliverance.toolcallparser.DefaultToolCallParser;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.mockito.Mockito;

import java.lang.reflect.Field;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.BiFunction;
import java.util.stream.Stream;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class Qwen3TensorParallelModelTest {
    @TempDir
    Path tempDir;

    @Test
    void tensorParallelRankLoadsLocalAttentionProjectionShards() {
        Qwen3Config config = tensorParallelTinyConfig();
        Path modelDir = Qwen3HfTextModelPortedTest.writeTinyCheckpoint(tempDir.resolve("qwen3-tp-shards"),
                config, 1234);
        try (Qwen3Model rank = loadRank(modelDir, 1, 2)) {
            CausalSelfAttention attention = attention(rank, 0);

            assertEquals(config.attentionLength / 2, tensor(attention, "queryAttnWeights").shape().first(),
                    "q_proj rows should be rank-local");
            assertEquals(config.kvLength / 2, tensor(attention, "keyAttnWeights").shape().first(),
                    "k_proj rows should be rank-local");
            assertEquals(config.kvLength / 2, tensor(attention, "valueAttnWeights").shape().first(),
                    "v_proj rows should be rank-local");
        }
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("tensorProviderCases")
    void tensorParallelForwardMatchesSingleModelForTinyQwen3(String providerName, ProviderFactory providerFactory) {
        if (providerName.equals("native-simd")) {
            Assumptions.assumeTrue(nativeSimdUsable(), "Native SIMD unavailable");
        }
        Qwen3Config config = tensorParallelTinyConfig();
        Path modelDir = Qwen3HfTextModelPortedTest.writeTinyCheckpoint(tempDir.resolve("qwen3-tp-forward-" + providerName),
                config, 2234);
        int[] tokens = new int[] {3, 4, 5, 6};
        InProcessTensorParallelCollectives.Group group = new InProcessTensorParallelCollectives.Group(Duration.ofSeconds(5));
        try (Qwen3Model single = loadRank(modelDir, 0, 1, providerFactory);
             Qwen3Model rank0 = loadRank(modelDir, 0, 2, group, providerFactory);
             Qwen3Model rank1 = loadRank(modelDir, 1, 2, group, providerFactory);
             TensorParallelGenerationGroup tp = new TensorParallelGenerationGroup(List.of(rank0, rank1));
             AbstractTensor singleOutput = single.batchForward(tokens, 0);
             AbstractTensor tpOutput = tp.batchForward(tokens, 0)) {
            try {
                assertTensorClose(providerName, singleOutput, tpOutput, 1.0e-4f);
            } catch (AssertionError e) {
                printTinyQwen3Diagnostics(providerName, modelDir, providerFactory, tokens);
                throw e;
            }
        }
    }

    @Test
    void printTinyQwen3PanamaSingleVsTensorParallelLayerZeroTensors() {
        Qwen3Config config = tensorParallelTinyConfig();
        Path modelDir = Qwen3HfTextModelPortedTest.writeTinyCheckpoint(tempDir.resolve("qwen3-tp-print"),
                config, 2234);
        ProviderFactory providerFactory = (allocator, pool) -> new ConfigurableTensorProvider(
                new PanamaTensorOperations(MachineSpec.VECTOR_TYPE, allocator, pool));
        int[] tokens = new int[] {3, 4, 5, 6};
        InProcessTensorParallelCollectives.Group group = new InProcessTensorParallelCollectives.Group(Duration.ofSeconds(5));
        try (Qwen3Model single = loadRank(modelDir, 0, 1, providerFactory);
             Qwen3Model rank0 = loadRank(modelDir, 0, 2, group, providerFactory);
             Qwen3Model rank1 = loadRank(modelDir, 1, 2, group, providerFactory);
             TensorParallelGenerationGroup tp = new TensorParallelGenerationGroup(List.of(rank0, rank1))) {
            AtomicReference<AbstractTensor> singleQuery = new AtomicReference<>();
            AtomicReference<AbstractTensor> rank0Query = new AtomicReference<>();
            AtomicReference<AbstractTensor> rank1Query = new AtomicReference<>();
            AtomicReference<AbstractTensor> singleAttentionOutput = new AtomicReference<>();
            AtomicReference<AbstractTensor> rank0AttentionOutput = new AtomicReference<>();

            single.setLayerDebugHook(event -> capture(event, "query_projection", singleQuery));
            single.setLayerDebugHook(event -> {
                capture(event, "query_projection", singleQuery);
                capture(event, "attention_output", singleAttentionOutput);
            });
            rank0.setLayerDebugHook(event -> {
                capture(event, "query_projection", rank0Query);
                capture(event, "attention_output", rank0AttentionOutput);
            });
            rank1.setLayerDebugHook(event -> capture(event, "query_projection", rank1Query));

            try (AbstractTensor singleOutput = single.batchForward(tokens, 0);
                 AbstractTensor tpOutput = tp.batchForward(tokens, 0)) {
                printTensor("single.layer0.query_projection", singleQuery.get());
                printTensor("rank0.layer0.query_projection", rank0Query.get());
                printTensor("rank1.layer0.query_projection", rank1Query.get());
                printTensor("single.layer0.attention_output", singleAttentionOutput.get());
                printTensor("rank0.layer0.attention_output", rank0AttentionOutput.get());
                printTensor("single.final", singleOutput);
                printTensor("tp.final", tpOutput);
                printDiff("final.diff", singleOutput, tpOutput);
            } finally {
                single.clearLayerDebugHook();
                rank0.clearLayerDebugHook();
                rank1.clearLayerDebugHook();
                close(singleQuery.get());
                close(rank0Query.get());
                close(rank1Query.get());
                close(singleAttentionOutput.get());
                close(rank0AttentionOutput.get());
            }
        }
    }

    private static void printTinyQwen3Diagnostics(String providerName, Path modelDir, ProviderFactory providerFactory,
            int[] tokens) {
        System.out.println("==== diagnostics provider=" + providerName + " ====");
        InProcessTensorParallelCollectives.Group group = new InProcessTensorParallelCollectives.Group(Duration.ofSeconds(5));
        try (Qwen3Model single = loadRank(modelDir, 0, 1, providerFactory);
             Qwen3Model rank0 = loadRank(modelDir, 0, 2, group, providerFactory);
             Qwen3Model rank1 = loadRank(modelDir, 1, 2, group, providerFactory);
             TensorParallelGenerationGroup tp = new TensorParallelGenerationGroup(List.of(rank0, rank1))) {
            AtomicReference<AbstractTensor> singleQuery = new AtomicReference<>();
            AtomicReference<AbstractTensor> rank0Query = new AtomicReference<>();
            AtomicReference<AbstractTensor> rank1Query = new AtomicReference<>();
            AtomicReference<AbstractTensor> singleAttentionOutput = new AtomicReference<>();
            AtomicReference<AbstractTensor> rank0AttentionOutput = new AtomicReference<>();
            single.setLayerDebugHook(event -> {
                capture(event, "query_projection", singleQuery);
                capture(event, "attention_output", singleAttentionOutput);
            });
            rank0.setLayerDebugHook(event -> {
                capture(event, "query_projection", rank0Query);
                capture(event, "attention_output", rank0AttentionOutput);
            });
            rank1.setLayerDebugHook(event -> capture(event, "query_projection", rank1Query));
            try (AbstractTensor singleOutput = single.batchForward(tokens, 0);
                 AbstractTensor tpOutput = tp.batchForward(tokens, 0)) {
                printTensor(providerName + ".single.layer0.query_projection", singleQuery.get());
                printTensor(providerName + ".rank0.layer0.query_projection", rank0Query.get());
                printTensor(providerName + ".rank1.layer0.query_projection", rank1Query.get());
                printTensor(providerName + ".single.layer0.attention_output", singleAttentionOutput.get());
                printTensor(providerName + ".rank0.layer0.attention_output", rank0AttentionOutput.get());
                printTensor(providerName + ".single.final", singleOutput);
                printTensor(providerName + ".tp.final", tpOutput);
                printDiff(providerName + ".final.diff", singleOutput, tpOutput);
            } finally {
                single.clearLayerDebugHook();
                rank0.clearLayerDebugHook();
                rank1.clearLayerDebugHook();
                close(singleQuery.get());
                close(rank0Query.get());
                close(rank1Query.get());
                close(singleAttentionOutput.get());
                close(rank0AttentionOutput.get());
            }
        }
    }

    private static Stream<Arguments> tensorProviderCases() {
        return Stream.of(
                Arguments.of("naive", (ProviderFactory) (allocator, pool) -> new ConfigurableTensorProvider(new NaiveTensorOperations())),
                Arguments.of("panama", (ProviderFactory) (allocator, pool) -> new ConfigurableTensorProvider(
                        new PanamaTensorOperations(MachineSpec.VECTOR_TYPE, allocator, pool))),
                Arguments.of("native-simd", (ProviderFactory) (allocator, pool) -> {
                    TensorOperations panama = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE, allocator, pool);
                    return new ConfigurableTensorProvider(new NativeSimdTensorOperations(panama));
                }));
    }

    private static Qwen3Config tensorParallelTinyConfig() {
        return new Qwen3Config(32, 16, 32, 4, 2, 2, 1.0e-6f, 64, null, 2,
                ActivationFunction.Type.SILU, 10_000.0, Map.of("rope_type", "default", "rope_theta", 10_000.0),
                4, false, null, 28, null, 0.0f, List.of("Qwen3ForCausalLM"));
    }

    private static Qwen3Model loadRank(Path modelDir, int rank, int size) {
        return loadRank(modelDir, rank, size,
                new InProcessTensorParallelCollectives.Group(Duration.ofSeconds(5)),
                (allocator, pool) -> new ConfigurableTensorProvider(new NaiveTensorOperations()));
    }

    private static Qwen3Model loadRank(Path modelDir, int rank, int size, ProviderFactory providerFactory) {
        return loadRank(modelDir, rank, size,
                new InProcessTensorParallelCollectives.Group(Duration.ofSeconds(5)), providerFactory);
    }

    private static Qwen3Model loadRank(Path modelDir, int rank, int size,
            InProcessTensorParallelCollectives.Group group) {
        return loadRank(modelDir, rank, size, group,
                (allocator, pool) -> new ConfigurableTensorProvider(new NaiveTensorOperations()));
    }

    private static Qwen3Model loadRank(Path modelDir, int rank, int size,
            InProcessTensorParallelCollectives.Group group, ProviderFactory providerFactory) {
        MetricRegistry metrics = new MetricRegistry();
        TensorAllocator allocator = new ArrayQueueTensorAllocator(metrics);
        WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores());
        StaticTensorParallelContext context = new StaticTensorParallelContext(rank, size);
        Qwen3Model model = new Qwen3Model(AbstractModel.InferenceType.FULL_GENERATION,
                Qwen3HfTextModelPortedTest.configFromFile(modelDir), new DefaultWeightLoader(modelDir.toFile()),
                Mockito.mock(io.teknek.deliverance.grace.PreTrainedTokenizer.class), DType.F32, DType.I8,
                Optional.empty(), providerFactory.create(allocator, pool), metrics, allocator,
                new KvBufferCacheSettings(true), new DefaultToolCallParser(), pool, context,
                size == 1
                        ? new io.teknek.deliverance.model.tensorparallel.SingleRankTensorParallelCollectives()
                        : new InProcessTensorParallelCollectives(context, group),
                Optional.empty());
        model.init();
        return model;
    }

    private static boolean nativeSimdUsable() {
        MetricRegistry metrics = new MetricRegistry();
        TensorAllocator allocator = new ArrayQueueTensorAllocator(metrics);
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new java.util.concurrent.ForkJoinPool(1));
             AbstractTensor input = TensorTestSupport.tensorOf(1, 4, 1, 2, 3, 4);
             AbstractTensor weight = TensorTestSupport.tensorOf(1, 4, 1, 1, 1, 1);
             AbstractTensor output = new io.teknek.deliverance.tensor.impl.FloatBufferTensor(1, 1)) {
            TensorOperations panama = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE, allocator, pool);
            TensorOperations nativeSimd = new NativeSimdTensorOperations(panama);
            nativeSimd.dotProductChunk(output, input, weight, 0, 4, 0, 1);
            return Math.abs(output.get(0, 0) - 10.0f) < 1.0e-6f;
        } catch (Throwable ignored) {
            return false;
        }
    }

    @FunctionalInterface
    private interface ProviderFactory {
        ConfigurableTensorProvider create(TensorAllocator allocator, WrappedForkJoinPool pool);
    }

    private static CausalSelfAttention attention(AbstractModel model, int layer) {
        TransformerBlock block = ((TransformerBlock[]) field(model, AbstractModel.class, "transformerBlocks"))[layer];
        return field(block, TransformerBlock.class, "attention");
    }

    private static AbstractTensor tensor(CausalSelfAttention attention, String name) {
        return field(attention, CausalSelfAttention.class, name);
    }

    private static void assertTensorClose(String label, AbstractTensor expected, AbstractTensor actual, float tolerance) {
        assertEquals(expected.shape().first(), actual.shape().first(), "row count");
        assertEquals(expected.shape().last(), actual.shape().last(), "column count");
        float max = 0.0f;
        int maxRow = -1;
        int maxCol = -1;
        for (int row = 0; row < expected.shape().first(); row++) {
            for (int col = 0; col < expected.shape().last(); col++) {
                float diff = Math.abs(expected.get(row, col) - actual.get(row, col));
                if (diff > max) {
                    max = diff;
                    maxRow = row;
                    maxCol = col;
                }
            }
        }
        int row = maxRow;
        int col = maxCol;
        float maxDiff = max;
        assertTrue(maxDiff <= tolerance, () -> label + " maxAbsDiff=" + maxDiff
                + " row=" + row
                + " col=" + col
                + " expected=" + expected.get(row, col)
                + " actual=" + actual.get(row, col));
    }

    private static void assertTensorClose(AbstractTensor expected, AbstractTensor actual, float tolerance) {
        assertTensorClose("tensor", expected, actual, tolerance);
    }

    private static void capture(AbstractModel.LayerDebugEvent event, String stage, AtomicReference<AbstractTensor> ref) {
        if (event.layerIndex() == 0 && event.stage().equals(stage)) {
            close(ref.getAndSet(new FloatBufferTensor(event.hiddenStates())));
        }
    }

    private static void printTensor(String label, AbstractTensor tensor) {
        System.out.println("==== " + label + " ====");
        System.out.println(tensor == null ? "<missing>" : TensorDisplayUtil.pretty2dDisplayAll(tensor));
    }

    private static void printDiff(String label, AbstractTensor expected, AbstractTensor actual) {
        FloatBufferTensor diff = new FloatBufferTensor(expected.shape().first(), expected.shape().last());
        try (diff) {
            for (int row = 0; row < expected.shape().first(); row++) {
                for (int col = 0; col < expected.shape().last(); col++) {
                    diff.set(actual.get(row, col) - expected.get(row, col), row, col);
                }
            }
            printTensor(label, diff);
        }
    }

    private static void close(AbstractTensor tensor) {
        if (tensor != null) {
            tensor.close();
        }
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
}
