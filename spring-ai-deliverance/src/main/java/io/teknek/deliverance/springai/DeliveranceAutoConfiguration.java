package io.teknek.deliverance.springai;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.teknek.deliverance.model.AutoModelConfig;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.model.CausalLanguageModel;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.core.io.Resource;
import org.springframework.core.io.ResourceLoader;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;

@AutoConfiguration
@EnableConfigurationProperties(DeliveranceConnectionProperties.class)
public class DeliveranceAutoConfiguration {

    @Bean
    @ConditionalOnMissingBean(ChatModel.class)
    @ConditionalOnProperty(prefix = "spring.ai.deliverance", name = "mode", havingValue = "client", matchIfMissing = true)
    public DeliveranceChatModel deliveranceChatModel(DeliveranceConnectionProperties properties, ObjectMapper objectMapper) {
        DeliveranceChatOptions options = DeliveranceChatOptions.builder()
                .model(properties.getModel())
                .build();
        return new DeliveranceChatModel(DeliveranceApi.create(properties.getBaseUrl(), properties.getApiKey()),
                objectMapper, options);
    }

    @Bean(destroyMethod = "close")
    @ConditionalOnMissingBean
    @ConditionalOnProperty(prefix = "spring.ai.deliverance", name = "mode", havingValue = "embedded")
    public CausalLanguageModel deliveranceCausalLanguageModel(DeliveranceConnectionProperties properties,
            ResourceLoader resourceLoader) {
        String model = properties.getModel();
        if (model == null || !model.contains("/")) {
            throw new IllegalArgumentException("Embedded mode requires spring.ai.deliverance.model in owner/name form");
        }
        if (properties.getHuggingface().getToken() != null && !properties.getHuggingface().getToken().isBlank()) {
            System.setProperty(ModelFetcher.HF_PROP, properties.getHuggingface().getToken());
        }
        String[] parts = model.split("/", 2);
        AutoModelForCausaLm.Builder builder = AutoModelForCausaLm.newBuilder(new ModelFetcher(parts[0], parts[1]))
                .withDownload(properties.isAutoPull());
        if (properties.getModelConfig() != null && !properties.getModelConfig().isBlank()) {
            builder.withConfig(AutoModelConfig.fromJson(resourceToPath(resourceLoader.getResource(properties.getModelConfig()))));
        }
        return builder.build();
    }

    @Bean
    @ConditionalOnMissingBean(ChatModel.class)
    @ConditionalOnProperty(prefix = "spring.ai.deliverance", name = "mode", havingValue = "embedded")
    public EmbeddedDeliveranceChatModel embeddedDeliveranceChatModel(CausalLanguageModel model,
            DeliveranceConnectionProperties properties) {
        return new EmbeddedDeliveranceChatModel(model, DeliveranceChatOptions.builder()
                .model(properties.getModel())
                .build());
    }

    private Path resourceToPath(Resource resource) {
        try {
            if (resource.isFile()) {
                return resource.getFile().toPath();
            }
            Path temp = Files.createTempFile("deliverance-model-config", ".json");
            try (var input = resource.getInputStream()) {
                Files.copy(input, temp, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
            }
            temp.toFile().deleteOnExit();
            return temp;
        } catch (IOException e) {
            throw new UncheckedIOException("Unable to read model config resource " + resource, e);
        }
    }
}
