
package io.teknek.reckoning.model.bert;


import io.teknek.reckoning.model.AbstractModel;
import io.teknek.reckoning.model.ModelType;
import io.teknek.reckoning.safetensors.Config;

public class BertModelType implements ModelType {
    @Override
    public Class<? extends AbstractModel> getModelClass() {
        return BertModel.class;
    }

    @Override
    public Class<? extends Config> getConfigClass() {
        return BertConfig.class;
    }

}
