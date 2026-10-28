package io.teknek.deliverance.model.bert;

import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.ModelType;
import io.teknek.deliverance.safetensors.Config;

public class XlmRobertaModelType implements ModelType {
    @Override
    public Class<? extends AbstractModel> getModelClass() {
        return XlmRobertaModel.class;
    }

    @Override
    public Class<? extends Config> getConfigClass() {
        return XlmRobertaConfig.class;
    }
}
