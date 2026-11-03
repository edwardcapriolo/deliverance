package net.deliverance.http;

import io.dropwizard.metrics5.MetricRegistry;

import io.teknek.deliverance.math.WrappedForkJoinPool;
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
    public WrappedForkJoinPool pool(@Value("${deliverance.pool-size:0}") int poolSize){
        if (poolSize > 0) {
            return new WrappedForkJoinPool(new ForkJoinPool(poolSize, ForkJoinPool.defaultForkJoinWorkerThreadFactory,
                    null, true));
        }
        return new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores());
    }

}
