
package io.teknek.deliverance.model.bert;


import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.ModelType;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.tokenizer.Tokenizer;

public class BertModelType implements ModelType {
    @Override
    public Class<? extends AbstractModel> getModelClass() {
        return BertModel.class;
    }

    @Override
    public Class<? extends Config> getConfigClass() {
        return BertConfig.class;
    }

    @Override
    public Class<? extends Tokenizer> getTokenizerClass() {
        return BertTokenizer.class;
    }
}