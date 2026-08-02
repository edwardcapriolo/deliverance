package io.teknek.deliverance.tensorlib;

import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorLocality;

import java.util.Optional;
import java.util.OptionalInt;

/** Best-effort native runtime hooks for CPU affinity and memory locality. */
public interface TensorRuntimeNative {
    boolean available();

    String reason();

    Optional<TensorLocality> localityOf(AbstractTensor tensor);

    OptionalInt currentCpu();

    OptionalInt currentNumaNode();

    OptionalInt numaNodeOfCpu(int cpu);

    default OptionalInt cpuForWorker(int workerIndex) {
        return OptionalInt.of(workerIndex);
    }

    boolean pinCurrentThread(int cpu);

    static TensorRuntimeNative unavailable(String reason) {
        return new UnavailableTensorRuntimeNative(reason);
    }

    final class UnavailableTensorRuntimeNative implements TensorRuntimeNative {
        private final String reason;

        private UnavailableTensorRuntimeNative(String reason) {
            this.reason = reason;
        }

        @Override
        public boolean available() {
            return false;
        }

        @Override
        public String reason() {
            return reason;
        }

        @Override
        public Optional<TensorLocality> localityOf(AbstractTensor tensor) {
            return Optional.empty();
        }

        @Override
        public OptionalInt currentCpu() {
            return OptionalInt.empty();
        }

        @Override
        public OptionalInt currentNumaNode() {
            return OptionalInt.empty();
        }

        @Override
        public OptionalInt numaNodeOfCpu(int cpu) {
            return OptionalInt.empty();
        }

        @Override
        public OptionalInt cpuForWorker(int workerIndex) {
            return OptionalInt.empty();
        }

        @Override
        public boolean pinCurrentThread(int cpu) {
            return false;
        }
    }
}
