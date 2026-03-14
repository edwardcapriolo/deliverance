package io.teknek.deliverance.model;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.model.qwen2.Qwen2TokenizerRenderer;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorCache;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;

import java.io.File;

public class AutoModelForCausaLm {

    public static AbstractModel fromPretrained(ModelFetcher fetcher){
        Builder b = new Builder(fetcher);
        if (fetcher.getName().startsWith("Llama")){
            b.withTokenTokenRenderer(new TokenizerRenderer());
        }
        if (fetcher.getName().startsWith("Qwen")){
            b.withTokenTokenRenderer(new Qwen2TokenizerRenderer());
        }
        return b.build();
    }

    public static Builder newBuilder(ModelFetcher fetcher){
        return new Builder(fetcher);
    }

    public static class Builder {
        private final ModelFetcher fetch;
        private MetricRegistry mr = new MetricRegistry();
        private TensorCache cache = new TensorCache(mr);
        private DType workingMem = DType.F32;
        private DType workingQuant = DType.I8;
        private TokenRenderer tokenRenderer = new NoOpTokenizerRenderer();

        private KvBufferCacheSettings settings = new KvBufferCacheSettings(true);
        private ConfigurableTensorProvider provider = new ConfigurableTensorProvider(cache);

        public Builder(ModelFetcher fetch){
            this.fetch = fetch;
        }

        public Builder withMetricRegistry(MetricRegistry metricRegistry){
            mr = metricRegistry;
            return this;
        }
        public Builder withTensorCache(TensorCache tensorCache){
            this.cache = tensorCache;
            return this;
        }
        public Builder withKvBufferCacheSettings(KvBufferCacheSettings settings){
            this.settings = settings;
            return this;
        }
        public Builder withWorkingMemoryType(DType type){
            this.workingMem = type;
            return this;
        }
        public Builder withWorkingQuantType(DType type){
            this.workingQuant = type;
            return this;
        }
        public Builder withTensorProvider(ConfigurableTensorProvider provider){
            this.provider = provider;
            return this;
        }
        public Builder withTokenTokenRenderer(TokenRenderer tokenRenderer){
            this.tokenRenderer = tokenRenderer;
            return this;
        }
        public AbstractModel build(){
            File modelRoot = fetch.maybeDownload();

            //ConfigurableTensorProvider jvmTensor = new ConfigurableTensorProvider(this.cache);
            //NativeSimdTensorOperations simd = new NativeSimdTensorOperations(jvmTensor.get());
            //ConfigurableTensorProvider top = new ConfigurableTensorProvider(simd);

            return ModelSupport.loadModel(modelRoot, workingMem, workingQuant, provider,
                    mr, cache, settings, fetch, tokenRenderer);
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
    }
}
