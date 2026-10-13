package io.teknek.deliverance.model;

import io.teknek.deliverance.safetensors.Config;

public interface ModelType {
    Class<? extends AbstractModel> getModelClass();

    Class<? extends Config> getConfigClass();

    /** Whether this model family should use legacy majority-dtype inference. */
    default boolean useMajorityDType() {
        return true;
    }
}
