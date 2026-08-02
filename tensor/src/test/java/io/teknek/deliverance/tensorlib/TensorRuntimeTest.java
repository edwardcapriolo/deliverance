package io.teknek.deliverance.tensorlib;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorLocality;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TensorRuntimeTest {

    @Test
    void analyzeModeRecordsNaturalLocality() {
        FakeNative nativeRuntime = new FakeNative();
        MetricRegistry metrics = new MetricRegistry();
        try (TensorRuntime runtime = new TensorRuntime(2, TensorRuntimeMode.ANALYZE, nativeRuntime, metrics);
             AbstractTensor tensor = new FloatBufferTensor(1, 32)) {
            tensor.setLocality(locality(tensor, 1));

            runtime.runAndWait("test", 0, Optional.of(tensor), () -> {});
            runtime.runAndWait("test", 1, Optional.of(tensor), () -> {});

            TensorRuntime.LocalitySnapshot snapshot = runtime.snapshot();
            assertEquals(2, snapshot.totalTasks());
            assertEquals(2, snapshot.local() + snapshot.remote() + snapshot.unknown());
        }
    }

    @Test
    void enforceModeChoosesWorkerOnTensorNumaNode() {
        FakeNative nativeRuntime = new FakeNative();
        MetricRegistry metrics = new MetricRegistry();
        AtomicInteger calls = new AtomicInteger();
        try (TensorRuntime runtime = new TensorRuntime(2, TensorRuntimeMode.ENFORCE, nativeRuntime, metrics);
             AbstractTensor tensor = new FloatBufferTensor(1, 32)) {
            tensor.setLocality(locality(tensor, 1));

            runtime.runAndWait("test", 0, Optional.of(tensor), calls::incrementAndGet);
            runtime.runAndWait("test", 1, Optional.of(tensor), calls::incrementAndGet);

            TensorRuntime.LocalitySnapshot snapshot = runtime.snapshot();
            assertEquals(2, calls.get());
            assertEquals(2, snapshot.totalTasks());
            assertEquals(2, snapshot.local());
            assertEquals(0, snapshot.remote());
            assertTrue(metrics.counter("tensorruntime.locality.local").getCount() >= 2);
        }
    }

    private static TensorLocality locality(AbstractTensor tensor, int numaNode) {
        return new TensorLocality(tensor.getMemorySegment().address(), tensor.getMemorySegment().byteSize(), numaNode,
                List.of(numaNode), 1L, "fake");
    }

    private static final class FakeNative implements TensorRuntimeNative {
        @Override
        public boolean available() {
            return true;
        }

        @Override
        public String reason() {
            return "fake";
        }

        @Override
        public Optional<TensorLocality> localityOf(AbstractTensor tensor) {
            return tensor.locality();
        }

        @Override
        public OptionalInt currentCpu() {
            return OptionalInt.of(0);
        }

        @Override
        public OptionalInt currentNumaNode() {
            return OptionalInt.of(0);
        }

        @Override
        public OptionalInt numaNodeOfCpu(int cpu) {
            return OptionalInt.of(cpu);
        }

        @Override
        public boolean pinCurrentThread(int cpu) {
            return true;
        }
    }
}
