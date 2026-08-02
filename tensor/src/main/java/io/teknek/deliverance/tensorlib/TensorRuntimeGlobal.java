package io.teknek.deliverance.tensorlib;

import com.codahale.metrics.MetricRegistry;

import java.util.Optional;

public final class TensorRuntimeGlobal {
    private static volatile TensorRuntime runtime;
    private static volatile TensorRuntimeNative nativeRuntime;

    private TensorRuntimeGlobal() {
    }

    public static TensorRuntime get(MetricRegistry metrics) {
        TensorRuntimeMode mode = TensorRuntimeMode.valueOf(
                System.getProperty("deliverance.tensor.runtime.mode", "DISABLED").trim().toUpperCase());
        int workers = Integer.getInteger("deliverance.tensor.runtime.workers",
                Math.max(1, Runtime.getRuntime().availableProcessors() / 2));
        return get(metrics, Optional.of(mode), workers);
    }

    public static TensorRuntime get(MetricRegistry metrics, Optional<TensorRuntimeMode> requestedMode, int workers) {
        TensorRuntimeMode mode = requestedMode.orElse(TensorRuntimeMode.DISABLED);
        if (mode == TensorRuntimeMode.DISABLED) {
            return null;
        }
        TensorRuntime current = runtime;
        if (current != null) {
            return current;
        }
        synchronized (TensorRuntimeGlobal.class) {
            if (runtime == null) {
                runtime = new TensorRuntime(Math.max(1, workers), mode, nativeRuntime(), metrics);
            }
            return runtime;
        }
    }

    private static TensorRuntimeNative nativeRuntime() {
        TensorRuntimeNative current = nativeRuntime;
        if (current != null) {
            return current;
        }
        try {
            current = (TensorRuntimeNative) Class.forName(
                    "io.teknek.deliverance.tensor.operations.runtime.NativeTensorRuntime")
                    .getConstructor()
                    .newInstance();
        } catch (ReflectiveOperationException | LinkageError e) {
            current = TensorRuntimeNative.unavailable("native runtime not available: " + e.getMessage());
        }
        nativeRuntime = current;
        return current;
    }
}
