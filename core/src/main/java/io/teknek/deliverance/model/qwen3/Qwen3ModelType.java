package io.teknek.deliverance.model.qwen3;

import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.ModelType;
import io.teknek.deliverance.safetensors.Config;

public class Qwen3ModelType implements ModelType {
    @Override
    public Class<? extends AbstractModel> getModelClass() {
        return Qwen3Model.class;
    }

    @Override
    public Class<? extends Config> getConfigClass() {
        return Qwen3Config.class;
    }
}
