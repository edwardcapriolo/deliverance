package io.teknek.deliverance.model.gemma4;

import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.ModelType;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.tokenizer.Tokenizer;

public class Gemma4ModelType implements ModelType {
    @Override
    public Class<? extends AbstractModel> getModelClass() {
        return Gemma4Model.class;
    }

    @Override
    public Class<? extends Config> getConfigClass() {
        return Gemma4Config.class;
    }

    @Override
    public Class<? extends Tokenizer> getTokenizerClass() {
        return Gemma4Tokenizer.class;
    }
}
