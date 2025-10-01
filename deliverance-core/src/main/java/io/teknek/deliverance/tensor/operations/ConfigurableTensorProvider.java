package io.teknek.deliverance.tensor.operations;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.atomic.AtomicReference;

public class ConfigurableTensorProvider {

    private static final Logger LOGGER = LoggerFactory.getLogger(TensorOperationsProvider.class);
    private final AtomicReference<TensorOperations> operations = new AtomicReference<>();

    public ConfigurableTensorProvider(){
        if (MachineSpec.VECTOR_TYPE == MachineSpec.Type.NONE){
            LOGGER.warn("Unable to determine vector type using NaiveTensorOperations");
            operations.set(new NaiveTensorOperations());
        } else {
            operations.set(new PanamaTensorOperations(MachineSpec.VECTOR_TYPE));
        }
    }

    public ConfigurableTensorProvider(TensorOperations userSupplied){
        operations.set(userSupplied);
    }

    public TensorOperations get(){
        return operations.get();
    }
}
