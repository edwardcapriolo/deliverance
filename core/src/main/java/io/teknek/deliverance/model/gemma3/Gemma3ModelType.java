package io.teknek.deliverance.model.gemma3;

import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.ModelType;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.tokenizer.Tokenizer;

public class Gemma3ModelType implements ModelType {
    @Override
    public Class<? extends AbstractModel> getModelClass() {
        return Gemma3Model.class;
    }

    @Override
    public Class<? extends Config> getConfigClass() {
        return Gemma3Config.class;
    }

    @Override
    public Class<? extends Tokenizer> getTokenizerClass() {
        return Gemma3Tokenizer.class;
    }
}