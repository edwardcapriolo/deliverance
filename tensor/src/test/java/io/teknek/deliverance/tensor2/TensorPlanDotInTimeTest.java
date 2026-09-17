package io.teknek.deliverance.tensor2;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorTestSupport;
import io.teknek.deliverance.tensor.operations.NaiveTensorOperations;
import io.teknek.deliverance.tensorlib.TensorPlan;
import io.teknek.deliverance.tensorlib.TensorPlanAdaptiveSplitTuner;
import io.teknek.dysfx.Either;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TensorPlanDotInTimeTest {

    @Test
    void scaleDotInTimeUsesSelectedProviderOnly() {
        CountingScaleOps simd = new CountingScaleOps(false);
        CountingScaleOps panama = new CountingScaleOps(true);
        CountingScaleOps naive = new CountingScaleOps(true);
        Lighter lighter = new Lighter(new MetricRegistry(), Map.of(
                TensorProviderKind.SIMD, simd,
                TensorProviderKind.PANAMA, panama,
                TensorProviderKind.NAIVE, naive
        ));
        FakeTuner tuner = new FakeTuner(TensorProviderKind.PANAMA.name());
        TensorPlan plan = new TensorPlan(new NaiveTensorOperations(), new WrappedForkJoinPool(new ForkJoinPool(1)),
                new MetricRegistry(), tuner);

        try (AbstractTensor logits = TensorTestSupport.tensorOf(1, 3, 2, 3, 4)) {
            plan.mutable("logits", logits)
                    .scale(10.0f)
                    .dotInTime("test.scale", lighter.providersFor(TensorProviderKind.SIMD, TensorProviderKind.PANAMA),
                            lighter.providerFor(TensorProviderKind.NAIVE))
                    .materialize();

            assertEquals(20.0f, logits.get(0, 0));
            assertEquals(30.0f, logits.get(0, 1));
            assertEquals(40.0f, logits.get(0, 2));
            assertEquals(0, simd.count());
            assertEquals(1, panama.count());
            assertEquals(0, naive.count());
            assertEquals("test.scale", tuner.observedPlanName());
            assertEquals(TensorProviderKind.PANAMA.name(), tuner.observedCandidate());
        }
    }

    @Test
    void scaleDotInTimeFallsBackWhenSelectedProviderDoesNotSupportOperation() {
        CountingScaleOps simd = new CountingScaleOps(false);
        CountingScaleOps panama = new CountingScaleOps(true);
        CountingScaleOps naive = new CountingScaleOps(true);
        Lighter lighter = new Lighter(new MetricRegistry(), Map.of(
                TensorProviderKind.SIMD, simd,
                TensorProviderKind.PANAMA, panama,
                TensorProviderKind.NAIVE, naive
        ));
        FakeTuner tuner = new FakeTuner(TensorProviderKind.SIMD.name());
        TensorPlan plan = new TensorPlan(new NaiveTensorOperations(), new WrappedForkJoinPool(new ForkJoinPool(1)),
                new MetricRegistry(), tuner);

        try (AbstractTensor logits = TensorTestSupport.tensorOf(1, 2, 5, 7)) {
            plan.mutable("logits", logits)
                    .scale(3.0f)
                    .dotInTime("test.scale", lighter.providersFor(TensorProviderKind.SIMD, TensorProviderKind.PANAMA),
                            lighter.providerFor(TensorProviderKind.NAIVE))
                    .materialize();

            assertEquals(15.0f, logits.get(0, 0));
            assertEquals(21.0f, logits.get(0, 1));
            assertEquals(1, simd.count());
            assertEquals(0, panama.count());
            assertEquals(1, naive.count());
            assertEquals(TensorProviderKind.SIMD.name(), tuner.observedCandidate());
            assertTrue(tuner.observedElapsedNanos() > 1_000_000_000_000L);
        }
    }

    @Test
    void scaleDotInTimeFallsBackWhenNoCandidatesAreRegistered() {
        CountingScaleOps naive = new CountingScaleOps(true);
        Lighter lighter = new Lighter(new MetricRegistry(), Map.of(TensorProviderKind.NAIVE, naive));
        FakeTuner tuner = new FakeTuner(TensorProviderKind.SIMD.name());
        TensorPlan plan = new TensorPlan(new NaiveTensorOperations(), new WrappedForkJoinPool(new ForkJoinPool(1)),
                new MetricRegistry(), tuner);

        try (AbstractTensor logits = TensorTestSupport.tensorOf(1, 2, 5, 7)) {
            plan.mutable("logits", logits)
                    .scale(3.0f)
                    .dotInTime("test.scale", lighter.providersFor(TensorProviderKind.SIMD, TensorProviderKind.PANAMA),
                            lighter.providerFor(TensorProviderKind.NAIVE))
                    .materialize();

            assertEquals(15.0f, logits.get(0, 0));
            assertEquals(21.0f, logits.get(0, 1));
            assertEquals(1, naive.count());
            assertFalse(tuner.chose());
        }
    }

    private static final class CountingScaleOps implements TensorOps {
        private final boolean supported;
        private final AtomicInteger count = new AtomicInteger();

        private CountingScaleOps(boolean supported) {
            this.supported = supported;
        }

        @Override
        public Either<OpSupport, Void> multiplyAccumulate(TensorRef a, TensorRef b, int offset, int length) {
            return Either.Left(OpSupport.Unsupported);
        }

        @Override
        public Either<OpSupport, Void> scale(float factor, TensorRef target, int offset, int length) {
            count.incrementAndGet();
            if (!supported) {
                return Either.Left(OpSupport.Unsupported);
            }
            for (int row = 0; row < target.shape().first(); row++) {
                for (int column = offset; column < offset + length; column++) {
                    target.underlying().set(target.underlying().get(row, column) * factor, row, column);
                }
            }
            return Either.Right(null);
        }

        private int count() {
            return count.get();
        }
    }

    private static final class FakeTuner implements TensorPlanAdaptiveSplitTuner {
        private final String selected;
        private boolean chose;
        private String observedPlanName;
        private String observedCandidate;
        private long observedElapsedNanos;

        private FakeTuner(String selected) {
            this.selected = selected;
        }

        @Override
        public int chooseSplit(String planName, io.teknek.deliverance.tensor.TensorShape shape, int defaultSplit,
                int minSplit, int maxSplit) {
            return defaultSplit;
        }

        @Override
        public void observeSplit(String planName, io.teknek.deliverance.tensor.TensorShape shape, int split,
                long elapsedNanos) {
        }

        @Override
        public String chooseAlternate(String planName, List<String> candidates) {
            chose = true;
            assertTrue(candidates.contains(selected));
            return selected;
        }

        @Override
        public void observeAlternate(String planName, String candidate, long elapsedNanos) {
            this.observedPlanName = planName;
            this.observedCandidate = candidate;
            this.observedElapsedNanos = elapsedNanos;
        }

        private boolean chose() {
            return chose;
        }

        private String observedPlanName() {
            return observedPlanName;
        }

        private String observedCandidate() {
            return observedCandidate;
        }

        private long observedElapsedNanos() {
            return observedElapsedNanos;
        }
    }
}
