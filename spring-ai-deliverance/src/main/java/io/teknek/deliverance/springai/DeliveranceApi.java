package io.teknek.deliverance.springai;

import java.time.Duration;

import com.fasterxml.jackson.annotation.JsonInclude;
import io.teknek.deliverance.client.spring.api.ChatApi;
import io.teknek.deliverance.client.spring.core.ApiClient;
import io.teknek.deliverance.client.spring.model.CreateChatCompletionRequest;
import io.teknek.deliverance.client.spring.model.CreateChatCompletionResponse;
import tools.jackson.databind.DeserializationFeature;
import tools.jackson.databind.json.JsonMapper;

public interface DeliveranceApi {
    CreateChatCompletionResponse createChatCompletion(CreateChatCompletionRequest request);

    static DeliveranceApi create(String baseUrl, String apiKey) {
        ApiClient apiClient = new ApiClient(jsonMapper(), ApiClient.createDefaultDateFormat()).setBasePath(baseUrl.endsWith("/")
                ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl);
        if (apiKey != null && !apiKey.isBlank()) {
            apiClient.addDefaultHeader("Authorization", "Bearer " + apiKey);
        }
        ChatApi chatApi = new ChatApi(apiClient);
        return request -> chatApi.createChatCompletion(request).block(Duration.ofMinutes(5));
    }

    static JsonMapper jsonMapper() {
        return JsonMapper.builder()
                .defaultDateFormat(ApiClient.createDefaultDateFormat())
                .changeDefaultPropertyInclusion(value -> JsonInclude.Value.construct(JsonInclude.Include.NON_NULL,
                        JsonInclude.Include.NON_NULL))
                .disable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES)
                .build();
    }
}
