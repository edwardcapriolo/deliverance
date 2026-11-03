package io.teknek.reckoning.model.llama;

import io.teknek.reckoning.model.AbstractModel;
import io.teknek.reckoning.model.ModelType;
import io.teknek.reckoning.safetensors.Config;

public class LlamaModelType implements ModelType {
    @Override
    public Class<? extends AbstractModel> getModelClass() {
        return LlamaModel.class;
    }

    @Override
    public Class<? extends Config> getConfigClass() {
        return LlamaConfig.class;
    }

}
