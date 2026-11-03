package io.teknek.reckoning.model.nemotronlabsdiffusion;

import io.teknek.reckoning.model.AbstractModel;
import io.teknek.reckoning.model.ModelType;
import io.teknek.reckoning.safetensors.Config;

public final class NemotronLabsDiffusionModelType implements ModelType {
    @Override
    public Class<? extends AbstractModel> getModelClass() {
        return NemotronLabsDiffusionModel.class;
    }

    @Override
    public Class<? extends Config> getConfigClass() {
        return NemotronLabsDiffusionConfig.class;
    }
}
