package net.deliverance.http;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;

import io.teknek.deliverance.generator.Generator;
import io.teknek.deliverance.model.ModelSupport;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorCache;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.io.File;

@Configuration
public class Config {

    @Bean
    public MetricRegistry metricRegistry(){
        return new MetricRegistry();
    }

    @Bean
    public Generator generator(@Value("${deliverance.model.name}") String modelName,
                               @Value("${deliverance.model.owner}") String modelOwner){
        //String modelName = "TinyLlama-1.1B-Chat-v1.0-Jlama-Q4";
        //String modelOwner = "tjake";
        ModelFetcher fetch = new ModelFetcher(modelOwner, modelName);
        File f = fetch.maybeDownload();

        TensorCache tensorCache = new TensorCache(metricRegistry());
        return ModelSupport.loadModel(f, DType.F32, DType.I8, new ConfigurableTensorProvider(tensorCache),
                metricRegistry(), tensorCache, new KvBufferCacheSettings(true));
    }
}
