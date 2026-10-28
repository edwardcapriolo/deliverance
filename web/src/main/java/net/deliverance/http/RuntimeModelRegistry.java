package net.deliverance.http;

import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.CausalLanguageModel;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@Configuration
public class RuntimeModelRegistry {

    @Bean(name = "causalLanguageModels")
    public Map<MultiModelConfig, CausalLanguageModel> causalLanguageModels() {
        return new ConcurrentHashMap<>();
    }

    @Bean(name = "embeddingModels")
    public Map<MultiModelConfig, AbstractModel> embeddingModels() {
        return new ConcurrentHashMap<>();
    }
}
