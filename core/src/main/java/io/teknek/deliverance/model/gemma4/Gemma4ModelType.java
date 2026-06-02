package io.teknek.deliverance.model.gemma4;

import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.ModelType;
import io.teknek.deliverance.safetensors.Config;

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
