package io.teknek.reckoning.model.gemma2;

import io.teknek.reckoning.model.AbstractModel;
import io.teknek.reckoning.model.ModelType;
import io.teknek.reckoning.safetensors.Config;

public class Gemma2ModelType implements ModelType {
    @Override
    public Class<? extends AbstractModel> getModelClass() {
        return Gemma2Model.class;
    }

    @Override
    public Class<? extends Config> getConfigClass() {
        return Gemma2Config.class;
    }

}
