package net.deliverance.http;

import io.dropwizard.metrics5.MetricRegistry;

import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.NativeGPUTensorOperations;
import io.teknek.deliverance.tensor.operations.NativeSimdTensorOperations;
import io.teknek.deliverance.tensor.operations.ParallelSplitSizedTensorOperations;
import io.teknek.deliverance.tensor.operations.TensorOperations;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.concurrent.ForkJoinPool;

@Configuration
public class Config {

    @Bean
    public MetricRegistry metricRegistry(){
        return new MetricRegistry();
    }

    @Bean
    public TensorAllocator tensorCache(){
        return new ArrayQueueTensorAllocator(metricRegistry());
    }

    @Bean
    public WrappedForkJoinPool pool(@Value("${deliverance.pool-size:0}") int poolSize){
        if (poolSize > 0) {
            return new WrappedForkJoinPool(new ForkJoinPool(poolSize, ForkJoinPool.defaultForkJoinWorkerThreadFactory,
                    null, true));
        }
        return new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores());
    }

    @Bean
    public ConfigurableTensorProvider provider(@Value("${deliverance.tensor.operations.type:simd}") String type,
            @Value("${deliverance.tensor.operations.simd.parallel-split-size:0}") int simdParallelSplitSize,
            WrappedForkJoinPool pool){

        if ("simd".equalsIgnoreCase(type)) {
            NativeSimdTensorOperations n = new NativeSimdTensorOperations(new ConfigurableTensorProvider(tensorCache(), pool).get());
            TensorOperations operations = simdParallelSplitSize > 0
                    ? new ParallelSplitSizedTensorOperations(n, simdParallelSplitSize)
                    : n;
            return new ConfigurableTensorProvider(operations);
        } else if ("jvector".equalsIgnoreCase(type)){
            return new ConfigurableTensorProvider(tensorCache(), pool);
        } else if ("gpu".equalsIgnoreCase(type)){
           NativeGPUTensorOperations g = new NativeGPUTensorOperations();
           return new ConfigurableTensorProvider(g);
        } else throw new IllegalArgumentException(type + " is not supported use (simd,jvector,gpu)");
    }

}
