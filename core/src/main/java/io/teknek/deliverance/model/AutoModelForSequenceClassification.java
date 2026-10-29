package io.teknek.deliverance.model;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.toolcallparser.ToolCallParser;
import io.dropwizard.metrics5.MetricRegistry;

/** Explicit-builder entry point for encoder sequence-classification models. */
public final class AutoModelForSequenceClassification {
    private AutoModelForSequenceClassification() {
    }

    public static AbstractModel fromPretrained(ModelFetcher fetcher) {
        return newBuilder(fetcher).build();
    }

    public static Builder newBuilder(ModelFetcher fetcher) {
        return new Builder(fetcher);
    }

    public static final class Builder {
        private final AutoModelForCausaLm.Builder delegate;

        public Builder(ModelFetcher fetcher) {
            delegate = AutoModelForCausaLm.newBuilder(fetcher);
        }

        public Builder withMetricRegistry(MetricRegistry value) {
            delegate.withMetricRegistry(value);
            return this;
        }

        /** @deprecated V1 allocator injection; use the future Lighter-based builder API. */
        @Deprecated(forRemoval = true)
        public Builder withTensorAllocator(TensorAllocator value) {
            delegate.withTensorAllocator(value);
            return this;
        }

        public Builder withKvBufferCacheSettings(KvBufferCacheSettings value) {
            delegate.withKvBufferCacheSettings(value);
            return this;
        }

        public Builder withWorkingMemoryType(DType value) {
            delegate.withWorkingMemoryType(value);
            return this;
        }

        public Builder withWorkingQuantType(DType value) {
            delegate.withWorkingQuantType(value);
            return this;
        }

        /** @deprecated V1 provider injection; use the future Lighter-based builder API. */
        @Deprecated(forRemoval = true)
        public Builder withTensorProvider(ConfigurableTensorProvider value) {
            delegate.withTensorProvider(value);
            return this;
        }

        public Builder withToolCallParser(ToolCallParser value) {
            delegate.withToolCallParser(value);
            return this;
        }

        public Builder withWrappedForkJoinPool(WrappedForkJoinPool value) {
            delegate.withWrappedForkJoinPool(value);
            return this;
        }

        public Builder withSystemPropertyForVectorOobCheck(String value) {
            delegate.withSystemPropertyForVectorOobCheck(value);
            return this;
        }

        public AbstractModel build() {
            return delegate.buildLocalTransformerModel(AbstractModel.InferenceType.FULL_CLASSIFICATION);
        }
    }
}
