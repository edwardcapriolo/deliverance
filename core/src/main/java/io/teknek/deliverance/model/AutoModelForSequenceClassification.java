package io.teknek.deliverance.model;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.TensorOperations;
import io.teknek.deliverance.toolcallparser.DefaultToolCallParser;
import io.teknek.deliverance.toolcallparser.ToolCallParser;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Optional;

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
        private TensorAllocator allocator = new ArrayQueueTensorAllocator(mr);
        private DType workingMem = DType.F32;
        private DType workingQuant = DType.I8;
        private ToolCallParser toolCallParser = new DefaultToolCallParser();

        private KvBufferCacheSettings settings = new KvBufferCacheSettings(true);
        private ConfigurableTensorProvider provider;
        private WrappedForkJoinPool pool;
        private String oobCheck = "2";

        public Builder(ModelFetcher fetch) {
            this.fetch = fetch;
        }

        public Builder withMetricRegistry(MetricRegistry metricRegistry) {
            mr = metricRegistry;
            return this;
        }

        public Builder withTensorAllocator(TensorAllocator arrayQueueTensorAllocator) {
            this.allocator = arrayQueueTensorAllocator;
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

        public Builder withToolCallParser(ToolCallParser toolCallParser) {
            this.toolCallParser = toolCallParser;
            return this;
        }

        public Builder withWrappedForkJoinPool(WrappedForkJoinPool pool){
            this.pool = pool;
            return this;
        }

        public Builder withSystemPropertyForVectorOobCheck(String value){
            this.oobCheck = value;
            return this;
        }

        public AbstractModel build() {
            System.setProperty("jdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK", this.oobCheck);
            File modelRoot = fetch.maybeDownload();
            if (pool == null){
                pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores());
            }
            if (provider == null){
                ConfigurableTensorProvider base = new ConfigurableTensorProvider(allocator, pool);
                Optional<TensorOperations> maybe = AutoModelForCausaLm.getNative(base.get());
                if (maybe.isPresent()){
                    provider = new ConfigurableTensorProvider(maybe.get());
                }
            }
            return ModelSupport.loadClassificationModel(modelRoot, workingMem, workingQuant, provider,
                    mr, allocator, settings, fetch, toolCallParser, pool);
        }

        public ModelFetcher getFetch() {
            return fetch;
        }

        public MetricRegistry getMr() {
            return mr;
        }

        public TensorAllocator getAllocator() {
            return allocator;
        }

        public DType getWorkingMem() {
            return workingMem;
        }

        public DType getWorkingQuant() {
            return workingQuant;
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
