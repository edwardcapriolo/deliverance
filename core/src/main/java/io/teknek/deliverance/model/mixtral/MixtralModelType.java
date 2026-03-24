package io.teknek.deliverance.model.mixtral;


import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.ModelType;
import io.teknek.deliverance.model.llama.LlamaTokenizer;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.tokenizer.Tokenizer;

public class MixtralModelType implements ModelType {
    @Override
    public Class<? extends AbstractModel> getModelClass() {
        return MixtralModel.class;
    }

    @Override
    public Class<? extends Config> getConfigClass() {
        return MixtralConfig.class;
    }

    @Override
    public Class<? extends Tokenizer> getTokenizerClass() {
        return LlamaTokenizer.class;
    }
}