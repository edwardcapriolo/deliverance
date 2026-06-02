package io.teknek.deliverance.model.mistral;

import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.ModelType;
import io.teknek.deliverance.safetensors.Config;

public class MistralModelType implements ModelType {
    @Override
    public Class<? extends AbstractModel> getModelClass() {
        return MistralModel.class;
    }

    @Override
    public Class<? extends Config> getConfigClass() {
        return MistralConfig.class;
    }

}
