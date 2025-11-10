package net.deliverance.http;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.ModelSupport;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorCache;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import org.springframework.boot.context.properties.ConfigurationProperties;
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
    private final TensorCache tensorCache;
    private final ConfigurableTensorProvider provider;

    public MultiModelConfiguration(MultiModelProperties multiModelProperties, MetricRegistry metricRegistry,
                                   TensorCache tensorCache,
                                   ConfigurableTensorProvider provider){
        this.multiModelProperties = multiModelProperties;
        this.metricRegistry = metricRegistry;
        this.tensorCache = tensorCache;
        this.provider = provider;
    }


    @Bean
    public Map<MultiModelConfig,AbstractModel> models(){
        Map<MultiModelConfig,AbstractModel> models = new HashMap<>();
        for (var x : multiModelProperties.getConfigs()){
            models.put(x, fromConfig(x));
        }
        return models;
    }


    private AbstractModel fromConfig(MultiModelConfig config){
        ModelFetcher fetch = new ModelFetcher(config.getModelOwner(),config.getModelName());
        File f = fetch.maybeDownload();
        if ("EMBEDDING".equalsIgnoreCase(config.getInferenceType())){
            AbstractModel model = ModelSupport.loadEmbeddingModel(f, DType.F32, DType.I8, provider,
                    metricRegistry, tensorCache, new KvBufferCacheSettings(true));
            return model;
        } else if ("GENERATION".equalsIgnoreCase(config.getInferenceType())){
            AbstractModel model = ModelSupport.loadModel(f, DType.F32, DType.I8, provider,
                    metricRegistry, tensorCache, new KvBufferCacheSettings(true));
            return model;
        } else {
            throw new IllegalArgumentException("Wrong type: " + config.getInferenceType());
        }

    }
}
