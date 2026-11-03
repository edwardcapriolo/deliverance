package io.teknek.reckoning.model;

import io.teknek.reckoning.safetensors.Config;

public interface ModelType {
    Class<? extends AbstractModel> getModelClass();

    Class<? extends Config> getConfigClass();

    /** Whether this model family should use legacy majority-dtype inference. */
    default boolean useMajorityDType() {
        return true;
    }
}
