package io.teknek.deliverance.springai;

import org.springframework.ai.chat.model.ChatModel;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import tools.jackson.databind.ObjectMapper;

@AutoConfiguration
@EnableConfigurationProperties(DeliveranceConnectionProperties.class)
public class DeliveranceAutoConfiguration {

    @Bean
    @ConditionalOnMissingBean(ChatModel.class)
    @ConditionalOnProperty(prefix = "spring.ai.deliverance", name = "mode", havingValue = "client", matchIfMissing = true)
    public DeliveranceChatModel deliveranceChatModel(DeliveranceConnectionProperties properties) {
        DeliveranceChatOptions options = DeliveranceChatOptions.builder()
                .model(properties.getModel())
                .build();
        return new DeliveranceChatModel(DeliveranceApi.create(properties.getBaseUrl(), properties.getApiKey()),
                new ObjectMapper(), options);
    }
}
