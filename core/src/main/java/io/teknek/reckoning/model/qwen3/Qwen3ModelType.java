package io.teknek.reckoning.model.qwen3;

import io.teknek.reckoning.model.AbstractModel;
import io.teknek.reckoning.model.ModelType;
import io.teknek.reckoning.safetensors.Config;

public class Qwen3ModelType implements ModelType {
    @Override
    public Class<? extends AbstractModel> getModelClass() {
        return Qwen3Model.class;
    }

    @Override
    public Class<? extends Config> getConfigClass() {
        return Qwen3Config.class;
    }
}
