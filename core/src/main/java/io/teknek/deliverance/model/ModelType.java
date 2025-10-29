package io.teknek.deliverance.model;

import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.tokenizer.Tokenizer;

public interface ModelType {
    Class<? extends AbstractModel> getModelClass();

    Class<? extends Config> getConfigClass();

    Class<? extends Tokenizer> getTokenizerClass();
}
