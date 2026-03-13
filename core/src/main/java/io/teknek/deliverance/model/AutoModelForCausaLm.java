package io.teknek.deliverance.model;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorCache;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;

import java.io.File;

public class AutoModelForCausaLm {

    public static AbstractModel fromPretrained(ModelFetcher fetcher){
        return new Builder(fetcher).build();
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
        public Builder tensorProvider(ConfigurableTensorProvider provider){
            this.provider = provider;
            return this;
        }
        public AbstractModel build(){
            File modelRoot = fetch.maybeDownload();

            //ConfigurableTensorProvider jvmTensor = new ConfigurableTensorProvider(this.cache);
            //NativeSimdTensorOperations simd = new NativeSimdTensorOperations(jvmTensor.get());
            //ConfigurableTensorProvider top = new ConfigurableTensorProvider(simd);

            return ModelSupport.loadModel(modelRoot, workingMem, workingQuant, provider,
                    new MetricRegistry(), cache, settings, fetch);
        }
    }
}
