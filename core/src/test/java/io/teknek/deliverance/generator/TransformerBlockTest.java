package io.teknek.deliverance.generator;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.KvBufferCache;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.NaiveTensorOperations;
import io.teknek.deliverance.tensor.TensorTestSupport;
import io.teknek.deliverance.tensorlib.PlannedTensor;
import io.teknek.deliverance.tensorlib.TensorPlan;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.List;
import java.util.Optional;
import java.util.concurrent.ForkJoinPool;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.when;

class TransformerBlockTest {

    @Test
    void applyResidualRangeAddsResidualWithOptionalMultiplier() {
        try (var target = TensorTestSupport.tensorOf(2, 3, 1, 2, 3, 4, 5, 6);
             var residual = TensorTestSupport.tensorOf(2, 3, 10, 20, 30, 40, 50, 60)) {

            TransformerBlock.applyResidualRange(target, residual, 0.5f, 1, 4);

            assertEquals(1.0f, target.get(0, 0), 1.0e-6f);
            assertEquals(21.0f, target.get(0, 1), 1.0e-6f);
            assertEquals(31.5f, target.get(0, 2), 1.0e-6f);
            assertEquals(42.0f, target.get(1, 0), 1.0e-6f);
            assertEquals(52.5f, target.get(1, 1), 1.0e-6f);
            assertEquals(6.0f, target.get(1, 2), 1.0e-6f);
        }
    }

    @Test
    void plannedForwardMatchesDirectForwardWithPostFeedForwardNorm() {
        Config config = new Config(16, 4, 8, 2, 2, 1,
                1.0e-6f, 32, 2, List.of(1), ActivationFunction.Type.GELU_PYTORCH_TANH, null, null);
        MetricRegistry metrics = new MetricRegistry();
        TensorAllocator allocator = new ArrayQueueTensorAllocator(metrics);
        ConfigurableTensorProvider provider = new ConfigurableTensorProvider(new NaiveTensorOperations());
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new ForkJoinPool(1));
             AbstractTensor input = TensorTestSupport.tensorOf(2, 4,
                     0.10f, 0.20f, -0.30f, 0.40f,
                     -0.50f, 0.60f, 0.70f, -0.80f)) {
            AbstractModel model = model(config, metrics, allocator, provider, pool);
            TransformerBlock directBlock = block(model, provider);
            TransformerBlock plannedBlock = block(model, provider);

            try (KvBufferCache cache = new KvBufferCache(model, new KvBufferCacheSettings(true));
                 KvBufferCache.KvBuffer directKv = cache.getEphemeralKvBuffer();
                 KvBufferCache.KvBuffer plannedKv = cache.getEphemeralKvBuffer();
                 AbstractTensor directInput = new FloatBufferTensor(input);
                 AbstractTensor plannedInput = new FloatBufferTensor(input);
                 AbstractTensor direct = directBlock.forward(directInput, 0, directKv, Optional.empty(), ForwardPhase.PREFILL)) {
                PlannedTensor planned = plannedBlock.forward(new PlannedTensor(plannedInput,
                                TensorPlanSupport.plan(model, provider.get()).input("input", plannedInput)),
                        0, plannedKv, Optional.empty(), ForwardPhase.PREFILL);
                assertTensorClose(direct, planned.tensor(), 0.0001f);
                planned.tensor().close();
            }
        }
    }

    private static TransformerBlock block(AbstractModel model, ConfigurableTensorProvider provider) {
        return new TransformerBlock(model, 0,
                Optional.of(new ScaleNorm(model, 1.10f)),
                new AddAttention(0.25f),
                Optional.of(new ScaleNorm(model, 0.90f)),
                Optional.of(new ScaleNorm(model, 1.20f)),
                new AddFeedForward(-0.15f),
                Optional.of(new ScaleNorm(model, 1.30f)),
                Optional.empty(),
                provider);
    }

    private static AbstractModel model(Config config, MetricRegistry metrics, TensorAllocator allocator,
            ConfigurableTensorProvider provider, WrappedForkJoinPool pool) {
        AbstractModel model = Mockito.mock(AbstractModel.class);
        when(model.getConfig()).thenReturn(config);
        when(model.getMetricRegistry()).thenReturn(metrics);
        when(model.getTensorAllocator()).thenReturn(allocator);
        when(model.getConfigurableTensorProvider()).thenReturn(provider);
        when(model.getPool()).thenReturn(pool);
        when(model.getWorkingDType()).thenReturn(DType.F32);
        when(model.getLocalKvLength()).thenReturn(config.kvLength);
        when(model.maybeQuantizeReadOnly(Mockito.any(AbstractTensor.class), Mockito.anyString()))
                .thenAnswer(invocation -> new FloatBufferTensor((AbstractTensor) invocation.getArgument(0)));
        return model;
    }

    private static void assertTensorClose(AbstractTensor expected, AbstractTensor actual, float tolerance) {
        assertEquals(expected.shape().first(), actual.shape().first(), "row count");
        assertEquals(expected.shape().last(), actual.shape().last(), "column count");
        for (int row = 0; row < expected.shape().first(); row++) {
            for (int col = 0; col < expected.shape().last(); col++) {
                assertEquals(expected.get(row, col), actual.get(row, col), tolerance,
                        "row=" + row + " col=" + col);
            }
        }
    }

    private static final class ScaleNorm extends LayerNorm {
        private final AbstractModel model;
        private final float scale;

        private ScaleNorm(AbstractModel model, float scale) {
            super(model, null, new FloatBufferTensor(1, model.getConfig().embeddingLength), model.getMetricRegistry());
            this.model = model;
            this.scale = scale;
        }

        @Override
        public AbstractTensor forward(AbstractTensor input) {
            AbstractTensor output = model.getTensorAllocator().get(DType.F32, input.shape());
            for (int row = 0; row < input.shape().first(); row++) {
                for (int col = 0; col < input.shape().last(); col++) {
                    output.set(input.get(row, col) * scale, row, col);
                }
            }
            return output;
        }

        @Override
        public PlannedTensor forward(PlannedTensor input) {
            AbstractTensor output = forward(input.tensor());
            return new PlannedTensor(output, TensorPlanSupport.plan(model, model.getConfigurableTensorProvider().get())
                    .input("scale_norm", input.plan(), output));
        }
    }

    private static final class AddAttention implements SelfAttention {
        private final float delta;

        private AddAttention(float delta) {
            this.delta = delta;
        }

        @Override
        public AbstractTensor forward(AbstractTensor input, int startPosition, KvBufferCache.KvBuffer kvMem,
                Optional<java.util.function.Consumer<List<AbstractTensor>>> tensorReducer) {
            AbstractTensor output = new FloatBufferTensor(input.shape());
            for (int row = 0; row < input.shape().first(); row++) {
                for (int col = 0; col < input.shape().last(); col++) {
                    output.set(input.get(row, col) + delta, row, col);
                }
            }
            return output;
        }
    }

    private static final class AddFeedForward implements FeedForward {
        private final float delta;

        private AddFeedForward(float delta) {
            this.delta = delta;
        }

        @Override
        public AbstractTensor forward(AbstractTensor input,
                Optional<java.util.function.Consumer<List<AbstractTensor>>> tensorReducer) {
            AbstractTensor output = new FloatBufferTensor(input.shape());
            for (int row = 0; row < input.shape().first(); row++) {
                for (int col = 0; col < input.shape().last(); col++) {
                    output.set(input.get(row, col) + delta, row, col);
                }
            }
            return output;
        }
    }
}
