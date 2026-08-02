package io.teknek.deliverance.tensor.operations.runtime;

import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

final class NativeRuntime {
    private static final SymbolLookup SYMBOL_LOOKUP = SymbolLookup.loaderLookup()
            .or(Linker.nativeLinker().defaultLookup());

    private static final MethodHandle CURRENT_CPU = handle("runtime_current_cpu",
            FunctionDescriptor.of(JAVA_INT));
    private static final MethodHandle NUMA_NODE_OF_CPU = handle("runtime_numa_node_of_cpu",
            FunctionDescriptor.of(JAVA_INT, JAVA_INT));
    private static final MethodHandle CPU_FOR_WORKER = handle("runtime_cpu_for_worker",
            FunctionDescriptor.of(JAVA_INT, JAVA_INT));
    private static final MethodHandle MEMORY_NUMA_NODE = handle("runtime_memory_numa_node",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG));
    private static final MethodHandle PIN_CURRENT_THREAD = handle("runtime_pin_current_thread",
            FunctionDescriptor.of(JAVA_INT, JAVA_INT));

    private NativeRuntime() {
    }

    static int currentCpu() {
        try {
            return (int) CURRENT_CPU.invokeExact();
        } catch (Throwable t) {
            return -1;
        }
    }

    static int numaNodeOfCpu(int cpu) {
        try {
            return (int) NUMA_NODE_OF_CPU.invokeExact(cpu);
        } catch (Throwable t) {
            return -1;
        }
    }

    static int cpuForWorker(int workerIndex) {
        try {
            return (int) CPU_FOR_WORKER.invokeExact(workerIndex);
        } catch (Throwable t) {
            return -1;
        }
    }

    static int memoryNumaNode(MemorySegment segment) {
        try {
            return (int) MEMORY_NUMA_NODE.invokeExact(segment, segment.byteSize());
        } catch (Throwable t) {
            return -1;
        }
    }

    static boolean pinCurrentThread(int cpu) {
        try {
            return (int) PIN_CURRENT_THREAD.invokeExact(cpu) == 1;
        } catch (Throwable t) {
            return false;
        }
    }

    private static MethodHandle handle(String name, FunctionDescriptor descriptor) {
        MemorySegment address = SYMBOL_LOOKUP.find(name)
                .orElseThrow(() -> new UnsatisfiedLinkError("unresolved symbol: " + name));
        return Linker.nativeLinker().downcallHandle(address, descriptor);
    }
}
