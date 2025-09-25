package io.teknek.deliverance.model.llama;

import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.ModelType;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.tokenizer.Tokenizer;

public class LlamaModelType implements ModelType {
    @Override
    public Class<? extends AbstractModel> getModelClass() {
        return LlamaModel.class;
    }

    @Override
    public Class<? extends Config> getConfigClass() {
        return LlamaConfig.class;
    }

    @Override
    public Class<? extends Tokenizer> getTokenizerClass() {
        return LlamaTokenizer.class;
    }

}

