package io.teknek.deliverance.model.diffusiongemma;

import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.ModelType;
import io.teknek.deliverance.safetensors.Config;

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
