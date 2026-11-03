package io.teknek.reckoning.model.mixtral;


import io.teknek.reckoning.model.AbstractModel;
import io.teknek.reckoning.model.ModelType;
import io.teknek.reckoning.safetensors.Config;

public class MixtralModelType implements ModelType {
    @Override
    public Class<? extends AbstractModel> getModelClass() {
        return MixtralModel.class;
    }

    @Override
    public Class<? extends Config> getConfigClass() {
        return MixtralConfig.class;
    }

}
