package io.teknek.deliverance.model.nemotronlabsdiffusion;

import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.ModelType;
import io.teknek.deliverance.safetensors.Config;

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
