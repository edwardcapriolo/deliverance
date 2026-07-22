package io.teknek.deliverance.model.granitemoehybrid;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.NaiveTensorOperations;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.Optional;

import static io.teknek.deliverance.tensor.TensorTestSupport.tensorOf;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;

public class GraniteMoeHybridSharedMlpTest {

    @Test
    public void tinyForwardMatchesHandComputedSharedMlpMath() {
        GraniteMoeHybridConfig config = new GraniteMoeHybridConfig(16, 2, 2, 1, 1, 1, 1.0e-5f,
                8, 1, 1, ActivationFunction.Type.SILU, 10_000.0, null, null, 0.0f, 1.0f,
                1.0f, 1.0f, 1.0f, 2, 0, 0, false, 0.01f, java.util.List.of("attention"),
                "rope", 1, 1, 2, 2, 4, 2, 16, true, false, java.util.List.of("GraniteMoeHybridForCausalLM"));
        try (Harness harness = new Harness(config);
             FloatBufferTensor input = tensorOf(1, 2, 1.0f, 2.0f);
             FloatBufferTensor inputLinear = tensorOf(4, 2,
                     1.0f, 0.0f,
                     0.0f, 1.0f,
                     3.0f, 0.0f,
                     0.0f, 4.0f);
             FloatBufferTensor outputLinear = tensorOf(2, 2,
                     1.0f, 0.0f,
                     0.0f, 1.0f)) {
            GraniteMoeHybridSharedMlp mlp = new GraniteMoeHybridSharedMlp(harness.model, config, inputLinear,
                    outputLinear, harness.provider);

            try (AbstractTensor output = mlp.forward(input, Optional.empty())) {
                assertEquals(2.1931758f, output.get(0, 0), 1.0e-5f);
                assertEquals(14.092754f, output.get(0, 1), 1.0e-5f);
            }
        }
    }

    @Test
    public void activationGateHelperMatchesTransformersFormula() {
        try (FloatBufferTensor projected = tensorOf(1, 4,
                1.0f, 2.0f,
                3.0f, 8.0f);
             FloatBufferTensor hidden = new FloatBufferTensor(1, 2)) {

            GraniteMoeHybridSharedMlp.applyActivationGate(projected, hidden, ActivationFunction.Type.SILU, 2);

            assertEquals(2.1931758f, hidden.get(0, 0), 1.0e-5f);
            assertEquals(14.092754f, hidden.get(0, 1), 1.0e-5f);
        }
    }

    private static final class Harness implements AutoCloseable {
        private final MetricRegistry metrics = new MetricRegistry();
        private final TensorAllocator allocator = new ArrayQueueTensorAllocator(metrics);
        private final WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores());
        private final ConfigurableTensorProvider provider = new ConfigurableTensorProvider(new NaiveTensorOperations());
        private final AbstractModel model = Mockito.mock(AbstractModel.class);

        private Harness(GraniteMoeHybridConfig config) {
            when(this.model.getConfig()).thenReturn(config);
            when(this.model.getMetricRegistry()).thenReturn(this.metrics);
            when(this.model.getTensorAllocator()).thenReturn(this.allocator);
            when(this.model.getWorkingDType()).thenReturn(DType.F32);
            when(this.model.getPool()).thenReturn(this.pool);
            when(this.model.maybeQuantize(any(AbstractTensor.class))).thenAnswer(invocation -> invocation.getArgument(0));
            when(this.model.makeTensor(Mockito.anyInt(), Mockito.anyInt()))
                    .thenAnswer(invocation -> this.allocator.get(DType.F32,
                            io.teknek.deliverance.tensor.TensorShape.of(invocation.getArgument(0), invocation.getArgument(1))));
        }

        @Override
        public void close() {
        }
    }
}
