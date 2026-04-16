package io.teknek.deliverance.model;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.qwen2.Qwen2TokenizerRenderer;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.NativeSimdTensorOperations;
import io.teknek.deliverance.toolcallparser.DefaultToolCallParser;
import io.teknek.deliverance.toolcallparser.LlamaToolCallParser;
import io.teknek.deliverance.toolcallparser.QwenToolCallParser;
import io.teknek.deliverance.toolcallparser.ToolCallParser;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

public class AutoModelForCausaLm {
    private static final Logger LOGGER = LoggerFactory.getLogger(AutoModelForCausaLm.class);
    public static void applyTuning(ModelFetcher fetcher, Builder b){
        if (fetcher.getName().startsWith("Llama")){
            b.withTokenTokenRenderer(new TokenizerRenderer());
            b.withToolCallParser(new LlamaToolCallParser());
        }
        if (fetcher.getName().startsWith("Qwen")){
            b.withTokenTokenRenderer(new Qwen2TokenizerRenderer());
            b.withToolCallParser(new QwenToolCallParser());
        }
    }

    public static AbstractModel fromPretrained(ModelFetcher fetcher){
        Builder b = new Builder(fetcher);
        applyTuning(fetcher, b);
        return b.build();
    }

    public static Builder newBuilder(ModelFetcher fetcher){
        //There is an argument to be made we shouldnt tune both sides
        Builder b = new Builder(fetcher);
        applyTuning(fetcher, b);
        return b;
    }


    public static class Builder {
        private final ModelFetcher fetch;
        private MetricRegistry mr = new MetricRegistry();
        private TensorAllocator allocator = new ArrayQueueTensorAllocator(mr);
        private DType workingMem = DType.F32;
        private DType workingQuant = DType.I8;
        private TokenRenderer tokenRenderer = new NoOpTokenizerRenderer();
        private ToolCallParser toolCallParser = new DefaultToolCallParser();

        private KvBufferCacheSettings settings = new KvBufferCacheSettings(true);
        private ConfigurableTensorProvider provider;
        private WrappedForkJoinPool pool;
        private String oobCheck = "2";

        public Builder(ModelFetcher fetch){
            this.fetch = fetch;
        }

        public Builder withMetricRegistry(MetricRegistry metricRegistry){
            mr = metricRegistry;
            return this;
        }
        public Builder withTensorAllocator(TensorAllocator tensorAllocator){
            this.allocator = tensorAllocator;
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
        public Builder withToolCallParser(ToolCallParser toolCallParser){
            this.toolCallParser = toolCallParser;
            return this;
        }
        public Builder withWrappedForkJoinPool(WrappedForkJoinPool pool){
            this.pool = pool;
            return this;
        }
        /** This is a JVM wide property! **/
        public Builder withSystemPropertyForVectorOobCheck(String value){
            this.oobCheck = value;
            return this;
        }

        public AbstractModel build(){
            System.setProperty("jdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK", this.oobCheck);
            File modelRoot = fetch.maybeDownload();
            if (pool == null){
                pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores());
            }
            if (provider == null){
                ConfigurableTensorProvider base  = new ConfigurableTensorProvider(allocator, pool);
                 try {
                     NativeSimdTensorOperations operations = new NativeSimdTensorOperations(base.get());
                     provider = new ConfigurableTensorProvider(operations);
                 } catch (UnsatisfiedLinkError e){
                     LOGGER.warn("unable to load native SIMD support", e);
                     provider = base;
                 }
            }
            return ModelSupport.loadModel(modelRoot, workingMem, workingQuant, provider,
                    mr, allocator, settings, fetch, tokenRenderer, toolCallParser, pool);
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

        public TokenRenderer getTokenRenderer() {
            return tokenRenderer;
        }

        public KvBufferCacheSettings getSettings() {
            return settings;
        }

        public ConfigurableTensorProvider getProvider() {
            return provider;
        }

        public WrappedForkJoinPool getPool() {
            return this.pool;
        }
    }
}
