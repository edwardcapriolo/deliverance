package io.teknek.reckoning.model.gpt2;

import io.teknek.reckoning.model.AbstractModel;
import io.teknek.reckoning.model.ModelType;
import io.teknek.reckoning.safetensors.Config;

public class Gpt2ModelType implements ModelType {
    @Override
    public Class<? extends AbstractModel> getModelClass() {
        return Gpt2Model.class;
    }

    @Override
    public Class<? extends Config> getConfigClass() {
        return Gpt2Config.class;
    }

}
