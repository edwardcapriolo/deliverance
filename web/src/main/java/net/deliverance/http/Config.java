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


    @Bean(destroyMethod = "close")
    public AbstractModel generator(@Value("${deliverance.model.name}") String modelName,
                                   @Value("${deliverance.model.owner}") String modelOwner,
                                   @Value("${deliverance.startup.test:true}") boolean test){
        ModelFetcher fetch = new ModelFetcher(modelOwner, modelName);
        File f = fetch.maybeDownload();

        TensorCache tensorCache = new TensorCache(metricRegistry());
        AbstractModel m =  ModelSupport.loadModel(f, DType.F32, DType.I8, new ConfigurableTensorProvider(tensorCache),
                metricRegistry(), tensorCache, new KvBufferCacheSettings(true));
        if(test) {
            PromptContext ctx;
            {
                PromptSupport ps = m.promptSupport().get();
                ctx = ps.builder().addSystemMessage("You are a chatbot that writes short correct responses.")
                        .addUserMessage("generate number 1 only once").build();
            }
            System.out.println(m.generate(UUID.randomUUID(), ctx, new GeneratorParameters().withNtokens(30), (x, t) -> {
            }));
        }
        return m;
    }
}
