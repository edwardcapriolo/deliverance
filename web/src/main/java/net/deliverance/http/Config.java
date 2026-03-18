package net.deliverance.http;

import com.codahale.metrics.MetricRegistry;

import io.teknek.deliverance.tensor.TensorCache;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.NativeGPUTensorOperations;
import io.teknek.deliverance.tensor.operations.NativeSimdTensorOperations;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

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
