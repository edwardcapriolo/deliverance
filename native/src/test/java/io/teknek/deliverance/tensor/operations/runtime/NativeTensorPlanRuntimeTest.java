package io.teknek.deliverance.tensor.operations.runtime;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.operations.NaiveTensorOperations;
import io.teknek.deliverance.tensorlib.TensorPlan;
import io.teknek.deliverance.tensorlib.TensorRuntime;
import io.teknek.deliverance.tensorlib.TensorRuntimeMode;
import org.junit.jupiter.api.Test;

import java.util.concurrent.ForkJoinPool;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class NativeTensorPlanRuntimeTest {

    @Test
    void tensorPlanMultiplyChainUsesRealNativeRuntimeLocalityCounters() {
        int multiplies = 8;
        int workers = 8;
        MetricRegistry metrics = new MetricRegistry();
        NativeTensorRuntime nativeRuntime = new NativeTensorRuntime();
        assertTrue(nativeRuntime.available(), nativeRuntime.reason());

        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new ForkJoinPool(workers));
             TensorRuntime runtime = new TensorRuntime(workers, TensorRuntimeMode.ENFORCE, nativeRuntime, metrics);
             FloatBufferTensor input = tensor(1, 8, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f);
             FloatBufferTensor multiplier = tensor(1, 8, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f)) {
            TensorPlan plan = new TensorPlan(new NaiveTensorOperations(), pool, metrics, runtime);
            TensorPlan.Tensor current = plan.input("input", input);
            for (int i = 0; i < multiplies; i++) {
                current = current.multiply(plan.input("multiplier-" + i, multiplier));
            }

            try (AbstractTensor out = current.materialize()) {
                assertEquals(256.0f, out.get(0, 0), 1.0e-6f);
                assertEquals(2048.0f, out.get(0, 7), 1.0e-6f);
            }

            TensorRuntime.LocalitySnapshot snapshot = runtime.snapshot();
            long pinned = metrics.counter("tensorruntime.affinity.pinned").getCount();
            long failed = metrics.counter("tensorruntime.affinity.failed").getCount();
            long unsupported = metrics.counter("tensorruntime.affinity.unsupported").getCount();
            System.out.printf("TensorRuntime native=%s locality: local=%d remote=%d unknown=%d tasks=%d pinned=%d failed=%d unsupported=%d%n",
                    nativeRuntime.reason(), snapshot.local(), snapshot.remote(), snapshot.unknown(),
                    snapshot.totalTasks(), pinned, failed, unsupported);

            assertEquals(multiplies, snapshot.totalTasks());
            assertEquals(multiplies, snapshot.local() + snapshot.remote() + snapshot.unknown());
            assertEquals(snapshot.local(), metrics.counter("tensorruntime.locality.local").getCount());
            assertEquals(snapshot.remote(), metrics.counter("tensorruntime.locality.remote").getCount());
            assertEquals(snapshot.unknown(), metrics.counter("tensorruntime.locality.unknown").getCount());
            assertEquals(workers, pinned, "native runtime should pin or apply native affinity/QoS to every worker");
            assertEquals(0, failed, "native runtime worker pinning should not fail");
            assertEquals(0, unsupported, "native runtime worker pinning should be supported");

            if (input.locality().isPresent()) {
                assertEquals(multiplies, snapshot.local(),
                        "known tensor NUMA locality should make ENFORCE mode run every multiply task locally");
                assertEquals(0, snapshot.remote(), "ENFORCE mode should not choose remote workers for known locality");
                assertEquals(0, snapshot.unknown(), "known tensor NUMA locality should not be counted unknown");
            } else {
                assertEquals(0, snapshot.local(), "without native memory locality support, local must not be faked");
                assertEquals(0, snapshot.remote(), "without native memory locality support, remote must not be faked");
                assertEquals(multiplies, snapshot.unknown(),
                        "without native memory locality support, all TensorPlan multiply tasks should be counted unknown");
            }
        }
    }

    private static FloatBufferTensor tensor(int rows, int cols, float... values) {
        FloatBufferTensor tensor = new FloatBufferTensor(rows, cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                tensor.set(values[row * cols + col], row, col);
            }
        }
        return tensor;
    }
}
