package io.teknek.reckoning.model.granitemoehybrid;

import io.teknek.reckoning.model.AbstractModel;
import io.teknek.reckoning.model.ModelType;
import io.teknek.reckoning.safetensors.Config;

public class GraniteMoeHybridModelType implements ModelType {

    @Override
    public Class<? extends AbstractModel> getModelClass() {
        return GraniteMoeHybridModel.class;
    }

    @Override
    public Class<? extends Config> getConfigClass() {
        return GraniteMoeHybridConfig.class;
    }

}
