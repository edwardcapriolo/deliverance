package io.teknek.deliverance.tensor.operations;

import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.tensor.TensorCacheIface;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.atomic.AtomicReference;

public class ConfigurableTensorProvider {

    private static final Logger LOGGER = LoggerFactory.getLogger(ConfigurableTensorProvider.class);
    private final AtomicReference<TensorOperations> operations = new AtomicReference<>();

    public ConfigurableTensorProvider(TensorCacheIface tensorCache, WrappedForkJoinPool pool){
        if (MachineSpec.VECTOR_TYPE == MachineSpec.Type.NONE){
            LOGGER.warn("Unable to determine vector type using NaiveTensorOperations");
            operations.set(new NaiveTensorOperations());
        } else {
            operations.set(new PanamaTensorOperations(MachineSpec.VECTOR_TYPE, tensorCache, pool));
        }
    }

    public ConfigurableTensorProvider(TensorOperations userSupplied){
        operations.set(userSupplied);
    }

    public TensorOperations get(){
        return operations.get();
    }
}
