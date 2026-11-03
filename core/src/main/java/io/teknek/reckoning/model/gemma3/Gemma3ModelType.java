package io.teknek.reckoning.model.gemma3;

import io.teknek.reckoning.model.AbstractModel;
import io.teknek.reckoning.model.ModelType;
import io.teknek.reckoning.safetensors.Config;

public class Gemma3ModelType implements ModelType {
    @Override
    public Class<? extends AbstractModel> getModelClass() {
        return Gemma3Model.class;
    }

    @Override
    public Class<? extends Config> getConfigClass() {
        return Gemma3Config.class;
    }

}
