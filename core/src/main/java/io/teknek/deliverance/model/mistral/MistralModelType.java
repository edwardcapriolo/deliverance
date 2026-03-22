package io.teknek.deliverance.model.mistral;

import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.ModelSupport;
import io.teknek.deliverance.model.ModelType;
import io.teknek.deliverance.model.llama.LlamaTokenizer;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.tokenizer.Tokenizer;

public class MistralModelType implements ModelType {
    @Override
    public Class<? extends AbstractModel> getModelClass() {
        return MistralModel.class;
    }

    @Override
    public Class<? extends Config> getConfigClass() {
        return MistralConfig.class;
    }

    @Override
    public Class<? extends Tokenizer> getTokenizerClass() {
        return LlamaTokenizer.class;
    }
}
