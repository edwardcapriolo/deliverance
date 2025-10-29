package io.teknek.deliverance.tensor.operations;


import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.tensor.TensorCache;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@Deprecated
public class TensorOperationsProvider {
    static {
        System.setProperty("jdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK", "0");
    }

    private static final Logger logger = LoggerFactory.getLogger(TensorOperationsProvider.class);
    private static final boolean forcePanama = Boolean.getBoolean("jlama.force_panama_tensor_operations");
    private static final boolean forceSimd = Boolean.getBoolean("jlama.force_simd_tensor_operations");

    private static final String lock = "lock";
    private static TensorOperationsProvider instance;

    public static TensorOperations get() {
        if (instance == null) {
            synchronized (lock) {
                if (instance == null) instance = new TensorOperationsProvider();
            }
        }

        return instance.provider;
    }

    private final TensorOperations provider;

    private TensorOperationsProvider() {
        this.provider = pickFastestImplementation();
    }

    private TensorOperations pickFastestImplementation() {

        TensorOperations pick = null;

        if (!forcePanama) {
            if (!forceSimd) {
                try {
                    Class<? extends TensorOperations> nativeClazz = (Class<? extends TensorOperations>) Class.forName(
                            "com.github.tjake.jlama.tensor.operations.NativeGPUTensorOperations"
                    );
                    pick = nativeClazz.getConstructor().newInstance();
                    // This will throw if no shared lib found
                } catch (Throwable t) {
                    logger.warn("Native GPU operations not available. Consider adding 'com.github.tjake:jlama-native' to the classpath");
                    logger.debug("Exception when loading native", t);
                }
            }

            if (pick == null) {
                try {
                    Class<? extends TensorOperations> nativeClazz = (Class<? extends TensorOperations>) Class.forName(
                            "com.github.tjake.jlama.tensor.operations.NativeSimdTensorOperations"
                    );
                    pick = nativeClazz.getConstructor().newInstance();
                } catch (Throwable t2) {
                    logger.warn("Native SIMD operations not available. Consider adding 'com.github.tjake:jlama-native' to the classpath");
                    logger.debug("Exception when loading native", t2);
                }
            }
        }

        if (pick == null) {
            pick = MachineSpec.VECTOR_TYPE == MachineSpec.Type.NONE
                    ? new NaiveTensorOperations()
                    : new PanamaTensorOperations(MachineSpec.VECTOR_TYPE,

                    new TensorCache(new MetricRegistry()));
        }

        logger.info("Using {} ({})", pick.name(), "OffHeap");
        return pick;
    }
}