package io.teknek.deliverance.model.granitemoehybrid;

import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.ModelType;
import io.teknek.deliverance.safetensors.Config;

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
