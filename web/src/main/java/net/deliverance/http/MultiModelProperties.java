package net.deliverance.http;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.model.CausalLanguageModel;
import io.teknek.deliverance.model.ModelSupport;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.stereotype.Component;

import java.io.File;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CopyOnWriteArrayList;

@Component
@ConfigurationProperties(prefix = "deliverance-model")
public class MultiModelProperties {

    private List<MultiModelConfig> configs = new CopyOnWriteArrayList<>();

    public List<MultiModelConfig> getConfigs() {
        return configs;
    }

    public void setConfigs(List<MultiModelConfig> configs) {
        this.configs = configs;
    }

}

@Configuration
class MultiModelConfiguration {

    private final MultiModelProperties multiModelProperties;
    private final MetricRegistry metricRegistry;
    private final TensorAllocator arrayQueueTensorAllocator;
    private final ConfigurableTensorProvider provider;
    private final WrappedForkJoinPool pool;
    private final String kvDiskDirectory;

    public MultiModelConfiguration(MultiModelProperties multiModelProperties, MetricRegistry metricRegistry,
                                    TensorAllocator arrayQueueTensorAllocator,
                                    ConfigurableTensorProvider provider,
                                    WrappedForkJoinPool pool,
                                    @Value("${deliverance.kv.disk-dir:}") String kvDiskDirectory){
        this.multiModelProperties = multiModelProperties;
        this.metricRegistry = metricRegistry;
        this.arrayQueueTensorAllocator = arrayQueueTensorAllocator;
        this.provider = provider;
        this.pool = pool;
        this.kvDiskDirectory = kvDiskDirectory;
    }


    @Bean
    public Map<MultiModelConfig, CausalLanguageModel> causalLanguageModels(){
        Map<MultiModelConfig, CausalLanguageModel> models = new HashMap<>();
        for (var x : multiModelProperties.getConfigs()){
            if ("GENERATION".equalsIgnoreCase(x.getInferenceType())) {
                models.put(x, causalLanguageModelFromConfig(x));
            }
        }
        return models;
    }

    @Bean
    public Map<MultiModelConfig, AbstractModel> embeddingModels(){
        Map<MultiModelConfig, AbstractModel> models = new HashMap<>();
        for (var x : multiModelProperties.getConfigs()){
            if ("EMBEDDING".equalsIgnoreCase(x.getInferenceType())) {
                models.put(x, embeddingModelFromConfig(x));
            }
        }
        return models;
    }

    private AbstractModel embeddingModelFromConfig(MultiModelConfig config){
        ModelFetcher fetch = new ModelFetcher(config.getModelOwner(),config.getModelName());
        File f = fetch.maybeDownload();
        return ModelSupport.loadEmbeddingModel(f, DType.F32, DType.I8, provider,
                metricRegistry, this.arrayQueueTensorAllocator, kvBufferCacheSettings());
    }

    private CausalLanguageModel causalLanguageModelFromConfig(MultiModelConfig config){
        ModelFetcher fetch = new ModelFetcher(config.getModelOwner(),config.getModelName());
        return AutoModelForCausaLm.newBuilder(fetch)
                .withMetricRegistry(metricRegistry)
                .withTensorAllocator(arrayQueueTensorAllocator)
                .withTensorProvider(provider)
                .withWrappedForkJoinPool(pool)
                .withKvBufferCacheSettings(kvBufferCacheSettings())
                .build();
    }

    private KvBufferCacheSettings kvBufferCacheSettings() {
        if (kvDiskDirectory == null || kvDiskDirectory.isBlank()) {
            return new KvBufferCacheSettings(true);
        }
        return new KvBufferCacheSettings(new File(kvDiskDirectory));
    }
}
