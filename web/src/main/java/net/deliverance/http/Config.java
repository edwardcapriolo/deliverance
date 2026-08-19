package net.deliverance.http;

import com.codahale.metrics.MetricRegistry;

import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.NativeGPUTensorOperations;
import io.teknek.deliverance.tensor.operations.NativeSimdTensorOperations;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.concurrent.ForkJoinPool;

@Configuration
public class Config {
    private static volatile Integer localLauncherPoolSize;

    public static void useLocalLauncherPoolSize(int poolSize) {
        if (poolSize < 1) {
            throw new IllegalArgumentException("poolSize must be >= 1");
        }
        localLauncherPoolSize = poolSize;
    }

    @Bean
    public MetricRegistry metricRegistry(){
        return new MetricRegistry();
    }

    @Bean
    public TensorAllocator tensorCache(){
        return new ArrayQueueTensorAllocator(metricRegistry());
    }

    @Bean
    public WrappedForkJoinPool pool(){
        WrappedForkJoinPool pool = localLauncherPoolSize == null
                ? new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores())
                : new WrappedForkJoinPool(new ForkJoinPool(localLauncherPoolSize,
                        ForkJoinPool.defaultForkJoinWorkerThreadFactory, null, true));
        return pool;
    }

    @Bean
    public ConfigurableTensorProvider provider(@Value("${deliverance.tensor.operations.type:simd}") String type){

        if ("simd".equalsIgnoreCase(type)) {
            NativeSimdTensorOperations n = new NativeSimdTensorOperations(new ConfigurableTensorProvider(tensorCache(), pool()).get());
            return new ConfigurableTensorProvider(n);
        } else if ("jvector".equalsIgnoreCase(type)){
            return new ConfigurableTensorProvider(tensorCache(), pool());
        } else if ("gpu".equalsIgnoreCase(type)){
           NativeGPUTensorOperations g = new NativeGPUTensorOperations();
           return new ConfigurableTensorProvider(g);
        } else throw new IllegalArgumentException(type + " is not supported use (simd,jvector,gpu)");
    }

}
