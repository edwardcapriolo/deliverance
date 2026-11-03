package io.teknek.reckoning.model;

import io.teknek.reckoning.DType;
import io.teknek.reckoning.grace.PreTrainedTokenizer;
import io.teknek.reckoning.safetensors.Config;
import io.teknek.reckoning.safetensors.WeightLoader;
import io.teknek.reckoning.safetensors.fetch.ModelFetcher;
import io.teknek.reckoning.safetensors.fetch.LocalModelFetcher;

import java.nio.file.Path;

public final class AutoModelForEmbeddings {
    private AutoModelForEmbeddings() {
    }

    public static Builder newBuilder(ModelFetcher fetcher) {
        return new Builder(fetcher);
    }

    public static Builder newBuilder(Path modelDirectory) {
        return new Builder(new LocalModelFetcher(modelDirectory));
    }

    public static class Builder extends AutoModelForCausaLm.Builder {
        public Builder(ModelFetcher fetch) {
            super(fetch);
        }

        public AbstractModel buildLocalEmbeddingModel() {
            return loadLocalTransformerModel();
        }

        @Override
        public Builder withQuantizeOnDemand(DType targetType, String outputOwner, String outputModel) {
            super.withQuantizeOnDemand(targetType, outputOwner, outputModel);
            return this;
        }

        public Builder quantizeOnDemand(DType targetType, String outputOwner, String outputModel) {
            return withQuantizeOnDemand(targetType, outputOwner, outputModel);
        }

        @Override
        protected AbstractModel newModel(String modelType, AbstractModel.InferenceType inferenceType, Config config,
                WeightLoader weightLoader, PreTrainedTokenizer tokenizer) {
            return super.newModel(modelType, AbstractModel.InferenceType.FULL_EMBEDDING, config, weightLoader, tokenizer);
        }
    }
}
