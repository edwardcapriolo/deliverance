package net.deliverance.http;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;

import io.teknek.deliverance.generator.Generator;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.ModelSupport;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorCache;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.NativeGPUTensorOperations;
import io.teknek.deliverance.tensor.operations.NativeSimdTensorOperations;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.io.File;
import java.util.UUID;

@Configuration
public class Config {

    @Bean
    public MetricRegistry metricRegistry(){
        return new MetricRegistry();
    }

    @Bean
    public TensorCache tensorCache(){
        return new TensorCache(metricRegistry());
    }

    @Bean
    public ConfigurableTensorProvider provider(@Value("${deliverance.tensor.operations.type:simd}") String type){
        if ("simd".equalsIgnoreCase(type)) {
            NativeSimdTensorOperations n = new NativeSimdTensorOperations(new ConfigurableTensorProvider(tensorCache()).get());
            return new ConfigurableTensorProvider(n);
        } else if ("jvector".equalsIgnoreCase(type)){
            return new ConfigurableTensorProvider(tensorCache());
        } else if ("gpu".equalsIgnoreCase(type)){
           NativeGPUTensorOperations g = new NativeGPUTensorOperations();
           return new ConfigurableTensorProvider(g);
        } else throw new IllegalArgumentException(type + " is not supported use (simd,jvector,gpu)");
    }

}
