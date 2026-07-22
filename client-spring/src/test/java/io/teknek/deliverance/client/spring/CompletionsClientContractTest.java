package io.teknek.deliverance.client.spring;

import com.github.tomakehurst.wiremock.WireMockServer;
import io.teknek.deliverance.client.spring.api.CompletionsApi;
import io.teknek.deliverance.client.spring.core.ApiClient;
import io.teknek.deliverance.client.spring.model.CreateCompletionRequest;
import io.teknek.deliverance.client.spring.model.CreateCompletionResponse;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.time.Duration;
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
    void springClientSendsAndReceivesLegacyCompletion() {
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

        ApiClient apiClient = new ApiClient().setBasePath(server.baseUrl());
        CompletionsApi api = new CompletionsApi(apiClient);

        CreateCompletionResponse response = api.createCompletion(request()).block(Duration.ofSeconds(5));

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
