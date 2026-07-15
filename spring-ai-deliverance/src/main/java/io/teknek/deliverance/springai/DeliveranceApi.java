package io.teknek.deliverance.springai;

import io.teknek.deliverance.client.api.ChatApi;
import io.teknek.deliverance.client.core.ApiClient;
import io.teknek.deliverance.client.model.CreateChatCompletionRequest;
import io.teknek.deliverance.client.model.CreateChatCompletionResponse;
import okhttp3.Interceptor;
import retrofit2.Response;

import java.io.IOException;

public interface DeliveranceApi {
    CreateChatCompletionResponse createChatCompletion(CreateChatCompletionRequest request);

    static DeliveranceApi create(String baseUrl, String apiKey) {
        ApiClient apiClient = new ApiClient();
        String normalized = baseUrl.endsWith("/") ? baseUrl : baseUrl + "/";
        apiClient.getAdapterBuilder().baseUrl(normalized);
        if (apiKey != null && !apiKey.isBlank()) {
            apiClient.getOkBuilder().addInterceptor((Interceptor) chain -> chain.proceed(chain.request()
                    .newBuilder()
                    .header("Authorization", "Bearer " + apiKey)
                    .build()));
        }
        ChatApi chatApi = apiClient.createService(ChatApi.class);
        return request -> {
            try {
                Response<CreateChatCompletionResponse> response = chatApi.createChatCompletion(request).execute();
                if (!response.isSuccessful()) {
                    throw new IllegalStateException("Deliverance chat completion failed status=" + response.code());
                }
                return response.body();
            } catch (IOException e) {
                throw new IllegalStateException("Deliverance chat completion request failed", e);
            }
        };
    }
}
