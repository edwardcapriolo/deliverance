package io.teknek.deliverance.tensor2;

import io.dropwizard.metrics5.MetricName;
import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

class LighterTest {

    @Test
    void multiplyAccumulateUsesNaiveFallback() {
        Lighter lighter = new Lighter();
        TensorRef a = lighter.allocate(DType.F32, TensorShape.of(2, 3));
        TensorRef b = lighter.allocate(DType.F32, TensorShape.of(2, 3));

        a.underlying().set(2.0f, 0, 0);
        a.underlying().set(3.0f, 0, 1);
        a.underlying().set(4.0f, 1, 0);
        a.underlying().set(5.0f, 1, 1);
        b.underlying().set(10.0f, 0, 0);
        b.underlying().set(20.0f, 0, 1);
        b.underlying().set(30.0f, 1, 0);
        b.underlying().set(40.0f, 1, 1);

        lighter.multiplyAccumulate(new MultiplyAccumulate(b).into(a).offsetAndLength(0, 2));

        assertEquals(20.0f, a.underlying().get(0, 0));
        assertEquals(60.0f, a.underlying().get(0, 1));
        assertEquals(120.0f, a.underlying().get(1, 0));
        assertEquals(200.0f, a.underlying().get(1, 1));
        assertEquals(0.0f, a.underlying().get(0, 2));
    }

    @Test
    void multiplyAccumulateBroadcastsSourceRow() {
        Lighter lighter = new Lighter();
        TensorRef a = lighter.allocate(DType.F32, TensorShape.of(2, 2));
        TensorRef b = lighter.allocate(DType.F32, TensorShape.of(1, 2));

        a.underlying().set(2.0f, 0, 0);
        a.underlying().set(3.0f, 0, 1);
        a.underlying().set(4.0f, 1, 0);
        a.underlying().set(5.0f, 1, 1);
        b.underlying().set(10.0f, 0, 0);
        b.underlying().set(20.0f, 0, 1);

        lighter.multiplyAccumulate(new MultiplyAccumulate(b).into(a).offsetAndLength(0, 2));

        assertEquals(20.0f, a.underlying().get(0, 0));
        assertEquals(60.0f, a.underlying().get(0, 1));
        assertEquals(40.0f, a.underlying().get(1, 0));
        assertEquals(100.0f, a.underlying().get(1, 1));
    }

    @Test
    void multiplyAccumulateRecordsMetrics() {
        MetricRegistry metricRegistry = new MetricRegistry();
        Lighter lighter = lighterWithUnsupportedSimd(metricRegistry);
        TensorRef a = lighter.allocate(DType.F32, TensorShape.of(1, 1));
        TensorRef b = lighter.allocate(DType.F32, TensorShape.of(1, 1));

        lighter.multiplyAccumulate(new MultiplyAccumulate(b).into(a).offsetAndLength(0, 1), Map.of("phase", "test"));

        assertEquals(1, metricRegistry.meter(new MetricName("tensor2.multiply_accumulate",
                Map.of("phase", "test", "ops", "SIMD"))).getCount());
        Map<String, String> tags = Map.of("phase", "test", "ops", "PANAMA");
        assertEquals(1, metricRegistry.meter(new MetricName("tensor2.multiply_accumulate", tags)).getCount());
        assertEquals(1, metricRegistry.timer(new MetricName("tensor2.multiply_accumulate.time", tags)).getCount());
        assertEquals(0, metricRegistry.meter(new MetricName("tensor2.multiply_accumulate",
                Map.of("phase", "test", "ops", "NAIVE"))).getCount());
    }

    @Test
    void multiplyAccumulateFallsBackToNaiveWhenPanamaDoesNotSupportDType() {
        MetricRegistry metricRegistry = new MetricRegistry();
        Lighter lighter = lighterWithUnsupportedSimd(metricRegistry);
        TensorRef f32A = lighter.allocate(DType.F32, TensorShape.of(1, 1));
        TensorRef f32B = lighter.allocate(DType.F32, TensorShape.of(1, 1));
        TensorRef a = new TensorRef(new TensorRefState(LeaseState.USED, null, f32A.underlying(), f32A.shape(),
                DType.BF16, f32A.stride(), f32A.device(), null));
        TensorRef b = new TensorRef(new TensorRefState(LeaseState.USED, null, f32B.underlying(), f32B.shape(),
                DType.BF16, f32B.stride(), f32B.device(), null));
        a.underlying().set(2.0f, 0, 0);
        b.underlying().set(3.0f, 0, 0);

        lighter.multiplyAccumulate(new MultiplyAccumulate(b).into(a).offsetAndLength(0, 1), Map.of("phase", "test"));

        assertEquals(6.0f, a.underlying().get(0, 0));
        assertEquals(1, metricRegistry.meter(new MetricName("tensor2.multiply_accumulate",
                Map.of("phase", "test", "ops", "SIMD"))).getCount());
        assertEquals(1, metricRegistry.meter(new MetricName("tensor2.multiply_accumulate",
                Map.of("phase", "test", "ops", "PANAMA"))).getCount());
        assertEquals(1, metricRegistry.meter(new MetricName("tensor2.multiply_accumulate",
                Map.of("phase", "test", "ops", "NAIVE"))).getCount());
    }

    @Test
    void scaleUsesPanama() {
        MetricRegistry metricRegistry = new MetricRegistry();
        Lighter lighter = lighterWithUnsupportedSimd(metricRegistry);
        TensorRef target = lighter.allocate(DType.F32, TensorShape.of(1, 3));
        target.underlying().set(2.0f, 0, 0);
        target.underlying().set(3.0f, 0, 1);
        target.underlying().set(4.0f, 0, 2);

        lighter.scale(new Scale(10.0f).target(target).offsetAndLength(1, 2), Map.of("phase", "test"));

        assertEquals(2.0f, target.underlying().get(0, 0));
        assertEquals(30.0f, target.underlying().get(0, 1));
        assertEquals(40.0f, target.underlying().get(0, 2));
        assertEquals(1, metricRegistry.meter(new MetricName("tensor2.scale",
                Map.of("phase", "test", "ops", "SIMD"))).getCount());
        assertEquals(1, metricRegistry.meter(new MetricName("tensor2.scale",
                Map.of("phase", "test", "ops", "PANAMA"))).getCount());
        assertEquals(0, metricRegistry.meter(new MetricName("tensor2.scale",
                Map.of("phase", "test", "ops", "NAIVE"))).getCount());
    }

    private static Lighter lighterWithUnsupportedSimd(MetricRegistry metricRegistry) {
        return new Lighter(metricRegistry, Map.of(
                TensorProviderKind.SIMD, (a, b, offset, length) -> io.teknek.dysfx.Either.Left(OpSupport.Unsupported),
                TensorProviderKind.PANAMA, new PanamaOps(),
                TensorProviderKind.NAIVE, new NaiveOps()
        ));
    }

}
