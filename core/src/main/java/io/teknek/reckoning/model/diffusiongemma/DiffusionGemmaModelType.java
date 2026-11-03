package io.teknek.reckoning.model.diffusiongemma;

import io.teknek.reckoning.model.AbstractModel;
import io.teknek.reckoning.model.ModelType;
import io.teknek.reckoning.safetensors.Config;

public final class DiffusionGemmaModelType implements ModelType {
    @Override
    public Class<? extends AbstractModel> getModelClass() {
        return DiffusionGemmaModel.class;
    }

    @Override
    public Class<? extends Config> getConfigClass() {
        return DiffusionGemmaConfig.class;
    }
}
