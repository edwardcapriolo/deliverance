package io.teknek.deliverance.model.embeddinggemma2;

import io.teknek.deliverance.model.ModelType;

/** Model registry entry for the text-capable EmbeddingGemma 2 checkpoint. */
public final class EmbeddingGemma2ModelType implements ModelType {
    @Override
    public Class<EmbeddingGemma2Model> getModelClass() {
        return EmbeddingGemma2Model.class;
    }

    @Override
    public Class<EmbeddingGemma2Config> getConfigClass() {
        return EmbeddingGemma2Config.class;
    }
}
