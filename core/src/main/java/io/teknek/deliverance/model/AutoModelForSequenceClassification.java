package io.teknek.deliverance.model;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorCache;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.NativeSimdTensorOperations;
import io.teknek.deliverance.toolcallparser.DefaultToolCallParser;
import io.teknek.deliverance.toolcallparser.ToolCallParser;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

public class AutoModelForSequenceClassification {

    private static final Logger LOGGER = LoggerFactory.getLogger(AutoModelForSequenceClassification.class);
    public static AbstractModel fromPretrained(ModelFetcher fetcher){
        return new AutoModelForSequenceClassification.Builder(fetcher).build();
    }

    public static Builder newBuilder(ModelFetcher fetcher){
        return new AutoModelForSequenceClassification.Builder(fetcher);
    }
    public static class Builder {
        private final ModelFetcher fetch;
        private MetricRegistry mr = new MetricRegistry();
        private TensorCache cache = new TensorCache(mr);
        private DType workingMem = DType.F32;
        private DType workingQuant = DType.I8;
        private TokenRenderer tokenRenderer = new NoOpTokenizerRenderer();
        private ToolCallParser toolCallParser = new DefaultToolCallParser();

        private KvBufferCacheSettings settings = new KvBufferCacheSettings(true);
        private ConfigurableTensorProvider provider;
        private WrappedForkJoinPool pool;

        public Builder(ModelFetcher fetch) {
            this.fetch = fetch;
        }

        public Builder withMetricRegistry(MetricRegistry metricRegistry) {
            mr = metricRegistry;
            return this;
        }

        public Builder withTensorCache(TensorCache tensorCache) {
            this.cache = tensorCache;
            return this;
        }

        public Builder withKvBufferCacheSettings(KvBufferCacheSettings settings) {
            this.settings = settings;
            return this;
        }

        public Builder withWorkingMemoryType(DType type) {
            this.workingMem = type;
            return this;
        }

        public Builder withWorkingQuantType(DType type) {
            this.workingQuant = type;
            return this;
        }

        public Builder withTensorProvider(ConfigurableTensorProvider provider) {
            this.provider = provider;
            return this;
        }

        public Builder withTokenTokenRenderer(TokenRenderer tokenRenderer) {
            this.tokenRenderer = tokenRenderer;
            return this;
        }

        public Builder withToolCallParser(ToolCallParser toolCallParser) {
            this.toolCallParser = toolCallParser;
            return this;
        }

        public Builder withWrappedForkJoinPool(WrappedForkJoinPool pool){
            this.pool = pool;
            return this;
        }

        public AbstractModel build() {
            File modelRoot = fetch.maybeDownload();
            if (pool == null){
                pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores());
            }
            if (provider == null) {
                ConfigurableTensorProvider base = new ConfigurableTensorProvider(cache, pool);
                try {
                    NativeSimdTensorOperations operations = new NativeSimdTensorOperations(base.get());
                    provider = new ConfigurableTensorProvider(operations);
                } catch (UnsatisfiedLinkError e) {
                    LOGGER.warn("unable to load native SIMD support", e);
                    provider = base;
                }
            }
            return ModelSupport.loadClassificationModel(modelRoot, workingMem, workingQuant, provider,
                    mr, cache, settings, fetch, tokenRenderer, toolCallParser, pool);
        }

        public ModelFetcher getFetch() {
            return fetch;
        }

        public MetricRegistry getMr() {
            return mr;
        }

        public TensorCache getCache() {
            return cache;
        }

        public DType getWorkingMem() {
            return workingMem;
        }

        public DType getWorkingQuant() {
            return workingQuant;
        }

        public TokenRenderer getTokenRenderer() {
            return tokenRenderer;
        }

        public KvBufferCacheSettings getSettings() {
            return settings;
        }

        public ConfigurableTensorProvider getProvider() {
            return provider;
        }

        public WrappedForkJoinPool getPool(){return pool; }

    }
}
