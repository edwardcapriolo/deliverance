package io.teknek.deliverance.tensor.operations.runtime;

import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorLocality;
import io.teknek.deliverance.tensor.operations.util.JarSupport;
import io.teknek.deliverance.tensorlib.TensorRuntimeNative;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Optional;
import java.util.OptionalInt;

public final class NativeTensorRuntime implements TensorRuntimeNative {
    private static final Logger LOGGER = LoggerFactory.getLogger(NativeTensorRuntime.class);
    private static volatile boolean loaded;
    private static volatile String reason;

    static {
        boolean loadedFromJar = JarSupport.maybeLoadLibrary("deliverance");
        if (loadedFromJar) {
            loaded = true;
        } else {
            try {
                System.loadLibrary("deliverance");
                loaded = true;
            } catch (UnsatisfiedLinkError e) {
                reason = e.getMessage();
                LOGGER.debug("Native runtime did not load", e);
            }
        }
    }

    @Override
    public boolean available() {
        return loaded;
    }

    @Override
    public String reason() {
        return loaded ? "native runtime loaded" : "native runtime unavailable: " + reason;
    }

    @Override
    public Optional<TensorLocality> localityOf(AbstractTensor tensor) {
        if (!loaded) {
            return Optional.empty();
        }
        int node = NativeRuntime.memoryNumaNode(tensor.getMemorySegment());
        if (node < 0) {
            return Optional.empty();
        }
        return Optional.of(new TensorLocality(tensor.getMemorySegment().address(), tensor.getMemorySegment().byteSize(),
                node, List.of(), System.currentTimeMillis(), "native-runtime"));
    }

    @Override
    public OptionalInt currentCpu() {
        if (!loaded) {
            return OptionalInt.empty();
        }
        int cpu = NativeRuntime.currentCpu();
        return cpu >= 0 ? OptionalInt.of(cpu) : OptionalInt.empty();
    }

    @Override
    public OptionalInt currentNumaNode() {
        OptionalInt cpu = currentCpu();
        if (cpu.isEmpty()) {
            return OptionalInt.empty();
        }
        return numaNodeOfCpu(cpu.getAsInt());
    }

    @Override
    public OptionalInt numaNodeOfCpu(int cpu) {
        if (!loaded) {
            return OptionalInt.empty();
        }
        int node = NativeRuntime.numaNodeOfCpu(cpu);
        return node >= 0 ? OptionalInt.of(node) : OptionalInt.empty();
    }

    @Override
    public OptionalInt cpuForWorker(int workerIndex) {
        if (!loaded) {
            return OptionalInt.empty();
        }
        int cpu = NativeRuntime.cpuForWorker(workerIndex);
        return cpu >= 0 ? OptionalInt.of(cpu) : OptionalInt.empty();
    }

    @Override
    public boolean pinCurrentThread(int cpu) {
        return loaded && NativeRuntime.pinCurrentThread(cpu);
    }
}
