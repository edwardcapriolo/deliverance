package io.teknek.deliverance.client;

import com.github.tomakehurst.wiremock.WireMockServer;
import io.teknek.deliverance.client.api.CompletionsApi;
import io.teknek.deliverance.client.core.ApiClient;
import io.teknek.deliverance.client.model.CreateCompletionRequest;
import io.teknek.deliverance.client.model.CreateCompletionResponse;
import okhttp3.OkHttpClient;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import retrofit2.Retrofit;
import retrofit2.converter.jackson.JacksonConverterFactory;
import retrofit2.converter.scalars.ScalarsConverterFactory;

import java.math.BigDecimal;
import java.util.List;

import static com.github.tomakehurst.wiremock.client.WireMock.aResponse;
import static com.github.tomakehurst.wiremock.client.WireMock.equalToJson;
import static com.github.tomakehurst.wiremock.client.WireMock.post;
import static com.github.tomakehurst.wiremock.client.WireMock.postRequestedFor;
import static com.github.tomakehurst.wiremock.client.WireMock.urlEqualTo;
import static org.junit.jupiter.api.Assertions.assertEquals;

class CompletionsClientContractTest {
    private WireMockServer server;

    @BeforeEach
    void startServer() {
        server = new WireMockServer(0);
        server.start();
    }

    @AfterEach
    void stopServer() {
        server.stop();
    }

    @Test
    void retrofitClientSendsAndReceivesLegacyCompletion() throws Exception {
        server.stubFor(post(urlEqualTo("/v1/completions"))
                .willReturn(aResponse()
                        .withHeader("Content-Type", "application/json")
                        .withBody("""
                                {
                                  "id": "cmpl-test",
                                  "object": "text_completion",
                                  "model": "test-model",
                                  "choices": [
                                    { "text": " vulnerable file", "index": 0, "finish_reason": "stop" }
                                  ]
                                }
                                """)));

        ApiClient apiClient = new ApiClient();
        apiClient.setAdapterBuilder(new Retrofit.Builder()
                .baseUrl(server.baseUrl() + "/")
                .client(new OkHttpClient.Builder().build())
                .addConverterFactory(ScalarsConverterFactory.create())
                .addConverterFactory(JacksonConverterFactory.create()));
        CompletionsApi api = apiClient.createService(CompletionsApi.class);

        CreateCompletionResponse response = api.createCompletion(request()).execute().body();

        assertEquals("cmpl-test", response.getId());
        assertEquals("text_completion", response.getObject().getValue());
        assertEquals("test-model", response.getModel());
        assertEquals(" vulnerable file", response.getChoices().get(0).getText());
        assertEquals("stop", response.getChoices().get(0).getFinishReason().getValue());
        server.verify(postRequestedFor(urlEqualTo("/v1/completions"))
                .withRequestBody(equalToJson(expectedRequestJson(), true, true)));
    }

    private static CreateCompletionRequest request() {
        return new CreateCompletionRequest()
                .model("test-model")
                .prompt("RAW ANTARES PROMPT")
                .maxTokens(8)
                .temperature(BigDecimal.valueOf(0.3))
                .topP(BigDecimal.ONE)
                .stop(List.of("<|end_of_text|>", "<|start_of_role|>"))
                .stream(false)
                .seed(123);
    }

    private static String expectedRequestJson() {
        return """
                {
                  "model": "test-model",
                  "prompt": "RAW ANTARES PROMPT",
                  "max_tokens": 8,
                  "seed": 123,
                  "stop": ["<|end_of_text|>", "<|start_of_role|>"],
                  "stream": false,
                  "temperature": 0.3,
                  "top_p": 1
                }
                """;
    }
}
