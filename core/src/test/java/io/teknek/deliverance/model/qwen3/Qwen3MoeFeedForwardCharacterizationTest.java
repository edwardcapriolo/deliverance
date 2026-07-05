package io.teknek.deliverance.model.qwen3;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.NaiveTensorOperations;
import io.teknek.deliverance.tensor.operations.TensorOperations;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static io.teknek.deliverance.tensor.TensorTestSupport.deterministicTensor;
import static io.teknek.deliverance.tensor.TensorTestSupport.tensorOf;
import static org.mockito.Mockito.when;

public class Qwen3MoeFeedForwardCharacterizationTest {

    @Test
    public void tinyForwardMatchesHandComputedExpertMath() {
        CountingTensorOperations ops = new CountingTensorOperations();
        Qwen3MoeFeedForward ff = feedForward(ops, false);
        try (FloatBufferTensor input = tensorOf(1, 2, 2.0f, 3.0f)) {

            try (AbstractTensor output = ff.forward(input, java.util.Optional.empty())) {
                assertEquals(103.57998f, output.get(0, 0), 1.0e-5f);
                assertEquals(128.01842f, output.get(0, 1), 1.0e-5f);
            }
        }
    }

    @Test
    public void routerSelectsTopKAndCanNormalizeSelectedWeights() throws Exception {
        CountingTensorOperations ops = new CountingTensorOperations();
        Qwen3MoeFeedForward ff = feedForward(ops, true);
        int[] selectedExperts = new int[2];
        float[] selectedWeights = new float[2];
        try (FloatBufferTensor input = tensorOf(1, 2, 1.0f, 0.0f)) {

            ff.route(input, 0, selectedExperts, selectedWeights);
        }

        java.util.Arrays.sort(selectedExperts);
        assertArrayEquals(new int[]{1, 2}, selectedExperts);
        assertEquals(1.0f, selectedWeights[0] + selectedWeights[1], 1.0e-5f);
    }

    @Test
    public void providerRouterProjectionMatchesScalarRouterProjection() {
        CountingTensorOperations scalarOps = new CountingTensorOperations();
        CountingTensorOperations providerOps = new CountingTensorOperations();
        Qwen3MoeFeedForward scalar = feedForward(scalarOps, true, Qwen3MoeFeedForward.SCALAR_EXECUTION);
        Qwen3MoeFeedForward provider = feedForward(providerOps, true,
                new Qwen3MoeFeedForward.ExecutionStrategy(true, true, true, false));
        int[] scalarExperts = new int[2];
        float[] scalarWeights = new float[2];
        int[] providerExperts = new int[2];
        float[] providerWeights = new float[2];

        try (FloatBufferTensor input = tensorOf(1, 2, 1.0f, 0.0f)) {
            scalar.route(input, 0, scalarExperts, scalarWeights);
            provider.route(input, 0, providerExperts, providerWeights);
        }

        java.util.Arrays.sort(scalarExperts);
        java.util.Arrays.sort(providerExperts);
        assertArrayEquals(scalarExperts, providerExperts);
        assertArrayEquals(scalarWeights, providerWeights, 1.0e-5f);
        assertTrue(providerOps.batchDotProductCalls > 0,
                "Provider router projection should use TensorOperations.batchDotProduct");
    }

    @Test
    public void providerExecutionMatchesScalarExecution() {
        CountingTensorOperations scalarOps = new CountingTensorOperations();
        CountingTensorOperations providerOps = new CountingTensorOperations();
        Qwen3MoeFeedForward scalar = feedForward(scalarOps, true, Qwen3MoeFeedForward.SCALAR_EXECUTION);
        Qwen3MoeFeedForward provider = feedForward(providerOps, true, Qwen3MoeFeedForward.PROVIDER_EXECUTION);

        try (FloatBufferTensor input = tensorOf(2, 2,
                2.0f, 3.0f,
                4.0f, 5.0f);
             AbstractTensor scalarOutput = scalar.forward(input, java.util.Optional.empty());
             AbstractTensor providerOutput = provider.forward(input, java.util.Optional.empty())) {
            assertTensorClose(scalarOutput, providerOutput, 1.0e-5f);
            assertTrue(providerOps.batchDotProductCalls > 0,
                    "Provider execution should use TensorOperations.batchDotProduct for router and expert projections");
        }
    }

    @Test
    public void providerExecutionAcceptsQuantizedInputRows() {
        CountingTensorOperations providerOps = new CountingTensorOperations();
        Qwen3MoeConfig config = quantizedInputConfig();
        Qwen3MoeFeedForward provider = feedForward(providerOps, true, Qwen3MoeFeedForward.PROVIDER_EXECUTION,
                config, new MetricRegistry());

        try (FloatBufferTensor denseInput = deterministicTensor(2, config.embeddingLength, 409);
             AbstractTensor quantizedInput = providerOps.quantize(denseInput, DType.I8, 0, denseInput.shape().last());
             AbstractTensor output = provider.forward(quantizedInput, java.util.Optional.empty())) {
            assertEquals(2, output.shape().first());
            assertEquals(config.embeddingLength, output.shape().last());
            assertTrue(Float.isFinite(output.get(0, 0)));
            assertTrue(Float.isFinite(output.get(1, 1)));
        }
    }

    @Test
    public void topKChoosesLargestExpertProbabilities() throws Exception {
        Qwen3MoeFeedForward ff = feedForward(new CountingTensorOperations(), false);
        int[] selectedExperts = new int[2];
        float[] selectedWeights = new float[2];

        ff.topK(new float[]{0.10f, 0.70f, 0.20f}, selectedExperts, selectedWeights);

        assertArrayEquals(new float[]{0.70f, 0.20f}, selectedWeights, 1.0e-6f);
        java.util.Arrays.sort(selectedExperts);
        assertArrayEquals(new int[]{1, 2}, selectedExperts);
    }

    @Test
    public void currentForwardBypassesProviderGemmForExpertComputation() {
        CountingTensorOperations ops = new CountingTensorOperations();
        Qwen3MoeFeedForward ff = feedForward(ops, false);
        try (FloatBufferTensor input = tensorOf(2, 2, 2.0f, 3.0f, 4.0f, 5.0f)) {

            try (AbstractTensor ignored = ff.forward(input, java.util.Optional.empty())) {
                assertEquals(0, ops.batchDotProductCalls,
                        "Current Qwen3 MoE expert path is scalar Java and does not use provider GEMM");
                assertEquals(0, ops.dotProductBatchChunkCalls,
                        "Current Qwen3 MoE expert path is scalar Java and does not use batched provider GEMM");
            }
        }
    }

    @Test
    public void currentImplementationUsesPerTokenRouting() {
        Qwen3MoeFeedForward.ExecutionStrategy strategy = feedForward(new CountingTensorOperations(), false)
                .executionStrategy();

        assertTrue(strategy.tokenByTokenRouting(),
                "Current Qwen3 MoE implementation processes batch rows one token at a time");
    }

    @Test
    public void currentImplementationUsesScalarExpertAccumulation() {
        Qwen3MoeFeedForward.ExecutionStrategy strategy = feedForward(new CountingTensorOperations(), false)
                .executionStrategy();

        assertTrue(strategy.scalarExpertAccumulation(),
                "Current Qwen3 MoE implementation does not express expert projection as provider GEMM");
        assertTrue(!strategy.providerRouterProjection(),
                "Current Qwen3 MoE implementation does not express router projection as provider GEMM");
        assertTrue(!strategy.providerExpertProjection(),
                "Current Qwen3 MoE implementation does not express expert projection as provider GEMM");
    }

    @Disabled("Manual microbenchmark for Qwen3 MoE feed-forward profiling; enable when comparing implementations.")
    @Test
    public void benchmarkCurrentQwen3MoeFeedForward() {
        CountingTensorOperations ops = new CountingTensorOperations();
        MetricRegistry metrics = new MetricRegistry();
        Qwen3MoeFeedForward ff = feedForward(ops, false, Qwen3MoeFeedForward.SCALAR_EXECUTION,
                benchmarkConfig(), metrics);
        try (FloatBufferTensor input = deterministicTensor(16, 128, 409)) {
            InferenceProfiler.setEnabled(true);
            InferenceProfiler.reset();
            long start = System.nanoTime();
            try (AbstractTensor ignored = ff.forward(input, java.util.Optional.empty())) {
                long elapsedNanos = System.nanoTime() - start;
                System.out.printf(java.util.Locale.ROOT,
                        "qwen3moe benchmark batch=%d hidden=%d experts=%d top_k=%d moe_intermediate=%d elapsed_ms=%.3f%n",
                        input.shape().first(), benchmarkConfig().embeddingLength, benchmarkConfig().numExperts,
                        benchmarkConfig().numExpertsPerToken, benchmarkConfig().moeIntermediateSize,
                        elapsedNanos / 1_000_000.0);
                InferenceProfiler.printSummary("qwen3moe benchmark", 20);
                for (String counter : java.util.List.of(Qwen3MoeFeedForward.COUNTER_ROUTE_SCALAR,
                        Qwen3MoeFeedForward.COUNTER_ROUTE_PROVIDER)) {
                    if (InferenceProfiler.shouldPrintCounter(counter)) {
                        System.out.println("[profile-counter] " + counter + " count=" + InferenceProfiler.counterValue(counter));
                    }
                }
            }
        }
    }

    private static Qwen3MoeFeedForward feedForward(CountingTensorOperations ops, boolean normTopkProb) {
        return feedForward(ops, normTopkProb, Qwen3MoeFeedForward.SCALAR_EXECUTION);
    }

    private static Qwen3MoeFeedForward feedForward(CountingTensorOperations ops, boolean normTopkProb,
            Qwen3MoeFeedForward.ExecutionStrategy executionStrategy) {
        Qwen3MoeConfig config = smallConfig(normTopkProb);
        AbstractModel model = Mockito.mock(AbstractModel.class);
        MetricRegistry metrics = new MetricRegistry();
        TensorAllocator allocator = new ArrayQueueTensorAllocator(metrics);
        when(model.getMetricRegistry()).thenReturn(metrics);
        when(model.getTensorAllocator()).thenReturn(allocator);
        when(model.makeTensor(Mockito.anyInt(), Mockito.anyInt()))
                .thenAnswer(invocation -> new FloatBufferTensor(
                        (Integer) invocation.getArgument(0), (Integer) invocation.getArgument(1)));
        when(model.maybeQuantize(Mockito.any(AbstractTensor.class)))
                .thenAnswer(invocation -> new FloatBufferTensor((AbstractTensor) invocation.getArgument(0)));

        FloatBufferTensor router = tensorOf(3, 2,
                0.0f, 0.0f,
                2.0f, 0.0f,
                1.0f, 0.0f);

        AbstractTensor[] gate = new AbstractTensor[3];
        AbstractTensor[] up = new AbstractTensor[3];
        AbstractTensor[] down = new AbstractTensor[3];
        for (int expert = 0; expert < 3; expert++) {
            gate[expert] = tensorOf(1, 2, 1.0f, 1.0f);
            up[expert] = tensorOf(1, 2, 2.0f, 2.0f);
            down[expert] = tensorOf(2, 1, expert + 1.0f, expert + 1.5f);
        }
        return new Qwen3MoeFeedForward(model, config, router, gate, up, down, new ConfigurableTensorProvider(ops),
                executionStrategy);
    }

    private static Qwen3MoeConfig smallConfig(boolean normTopkProb) {
        return new Qwen3MoeConfig(16, 2, 4, 1, 1, 1, 1.0e-6f, 8, null, 1,
                ActivationFunction.Type.SILU, 10_000.0, Map.of("rope_type", "default", "rope_theta", 10_000.0),
                2, false, null, null, null, 0.0f, List.of("Qwen3MoeForCausalLM"),
                1, 1, 2, 3, normTopkProb, false, 0.001f, List.of());
    }

    private static Qwen3MoeConfig benchmarkConfig() {
        return new Qwen3MoeConfig(256, 128, 256, 4, 2, 1, 1.0e-6f, 256, null, 1,
                ActivationFunction.Type.SILU, 10_000.0, Map.of("rope_type", "default", "rope_theta", 10_000.0),
                32, false, null, null, null, 0.0f, List.of("Qwen3MoeForCausalLM"),
                1, 64, 4, 16, false, false, 0.001f, List.of());
    }

    private static Qwen3MoeConfig quantizedInputConfig() {
        return new Qwen3MoeConfig(64, 32, 64, 4, 2, 1, 1.0e-6f, 64, null, 1,
                ActivationFunction.Type.SILU, 10_000.0, Map.of("rope_type", "default", "rope_theta", 10_000.0),
                8, false, null, null, null, 0.0f, List.of("Qwen3MoeForCausalLM"),
                1, 32, 2, 4, true, false, 0.001f, List.of());
    }

    private static Qwen3MoeFeedForward feedForward(CountingTensorOperations ops, boolean normTopkProb,
            Qwen3MoeFeedForward.ExecutionStrategy executionStrategy, Qwen3MoeConfig config, MetricRegistry metrics) {
        AbstractModel model = Mockito.mock(AbstractModel.class);
        TensorAllocator allocator = new ArrayQueueTensorAllocator(metrics);
        when(model.getMetricRegistry()).thenReturn(metrics);
        when(model.getTensorAllocator()).thenReturn(allocator);
        when(model.makeTensor(Mockito.anyInt(), Mockito.anyInt()))
                .thenAnswer(invocation -> new FloatBufferTensor(
                        (Integer) invocation.getArgument(0), (Integer) invocation.getArgument(1)));
        when(model.maybeQuantize(Mockito.any(AbstractTensor.class)))
                .thenAnswer(invocation -> new FloatBufferTensor((AbstractTensor) invocation.getArgument(0)));

        FloatBufferTensor router = deterministicTensor(config.numExperts, config.embeddingLength, 17);

        AbstractTensor[] gate = new AbstractTensor[config.numExperts];
        AbstractTensor[] up = new AbstractTensor[config.numExperts];
        AbstractTensor[] down = new AbstractTensor[config.numExperts];
        for (int expert = 0; expert < config.numExperts; expert++) {
            gate[expert] = deterministicTensor(config.moeIntermediateSize, config.embeddingLength, 101 + expert);
            up[expert] = deterministicTensor(config.moeIntermediateSize, config.embeddingLength, 211 + expert);
            down[expert] = deterministicTensor(config.embeddingLength, config.moeIntermediateSize, 307 + expert);
        }
        return new Qwen3MoeFeedForward(model, config, router, gate, up, down, new ConfigurableTensorProvider(ops),
                executionStrategy);
    }

    private static void assertTensorClose(AbstractTensor expected, AbstractTensor actual, float tolerance) {
        assertEquals(expected.shape().first(), actual.shape().first());
        assertEquals(expected.shape().last(), actual.shape().last());
        for (int row = 0; row < expected.shape().first(); row++) {
            for (int col = 0; col < expected.shape().last(); col++) {
                assertEquals(expected.get(row, col), actual.get(row, col), tolerance,
                        "row=" + row + " col=" + col);
            }
        }
    }

    private static class CountingTensorOperations implements TensorOperations {
        private final NaiveTensorOperations delegate = new NaiveTensorOperations();
        private int batchDotProductCalls;
        private int dotProductBatchChunkCalls;

        @Override
        public String name() {
            return "counting";
        }

        @Override
        public int parallelSplitSize() {
            return delegate.parallelSplitSize();
        }

        @Override
        public void batchDotProduct(AbstractTensor result, AbstractTensor a, AbstractTensor b, int aColumnOffset,
                int bColumnOffset, int columnLimit, int rRowOffset, int bRowOffset, int rowChunkSize) {
            batchDotProductCalls++;
            delegate.batchDotProduct(result, a, b, aColumnOffset, bColumnOffset, columnLimit,
                    rRowOffset, bRowOffset, rowChunkSize);
        }

        @Override
        public void dotProductBatchChunk(AbstractTensor[] result, AbstractTensor a, AbstractTensor[] b, int offset,
                int limit, int chunkStart, int chunkSize) {
            dotProductBatchChunkCalls++;
            delegate.dotProductBatchChunk(result, a, b, offset, limit, chunkStart, chunkSize);
        }

        @Override
        public void accumulate(AbstractTensor a, AbstractTensor b, int offset, int length) {
            delegate.accumulate(a, b, offset, length);
        }

        @Override
        public void maccumulate(AbstractTensor a, AbstractTensor b, int offset, int length) {
            delegate.maccumulate(a, b, offset, length);
        }

        @Override
        public void saxpy(float alpha, AbstractTensor x, AbstractTensor y, int xoffset, int yoffset, int limit) {
            delegate.saxpy(alpha, x, y, xoffset, yoffset, limit);
        }

        @Override
        public void scale(float factor, AbstractTensor x, int offset, int length) {
            delegate.scale(factor, x, offset, length);
        }

        @Override
        public AbstractTensor quantize(AbstractTensor t, DType qtype, int offset, int length) {
            return delegate.quantize(t, qtype, offset, length);
        }
    }
}
