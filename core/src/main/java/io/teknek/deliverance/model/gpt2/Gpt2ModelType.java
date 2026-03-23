package io.teknek.deliverance.model.gpt2;

import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.ModelType;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.tokenizer.Tokenizer;

public class Gpt2ModelType implements ModelType {
    @Override
    public Class<? extends AbstractModel> getModelClass() {
        //return GPT2Model.class;
        return null;
    }

    @Override
    public Class<? extends Config> getConfigClass() {
        return Gpt2Config.class;
    }

    @Override
    public Class<? extends Tokenizer> getTokenizerClass() {
        return Gpt2Tokenizer.class;
    }
}