package io.teknek.deliverance.model;

import io.teknek.deliverance.grace.PreTrainedTokenizer;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.safetensors.WeightLoader;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;

public final class AutoModelForEmbeddings {
    private AutoModelForEmbeddings() {
    }

    public static Builder newBuilder(ModelFetcher fetcher) {
        return new Builder(fetcher);
    }

    public static class Builder extends AutoModelForCausaLm.Builder {
        public Builder(ModelFetcher fetch) {
            super(fetch);
        }

        public AbstractModel buildLocalEmbeddingModel() {
            return loadLocalTransformerModel();
        }

        @Override
        protected AbstractModel newModel(String modelType, AbstractModel.InferenceType inferenceType, Config config,
                WeightLoader weightLoader, PreTrainedTokenizer tokenizer) {
            return super.newModel(modelType, AbstractModel.InferenceType.FULL_EMBEDDING, config, weightLoader, tokenizer);
        }
    }
}
