package io.teknek.reckoning.model.gemma4;

import io.teknek.reckoning.model.AbstractModel;
import io.teknek.reckoning.model.ModelType;
import io.teknek.reckoning.safetensors.Config;

public class Gemma4ModelType implements ModelType {
    @Override
    public Class<? extends AbstractModel> getModelClass() {
        return Gemma4Model.class;
    }

    @Override
    public Class<? extends Config> getConfigClass() {
        return Gemma4Config.class;
    }

}
