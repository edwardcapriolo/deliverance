package io.teknek.deliverance.model;

import io.teknek.deliverance.safetensors.Config;

public interface ModelType {
    Class<? extends AbstractModel> getModelClass();

    Class<? extends Config> getConfigClass();
}
