package io.teknek.deliverance.nanocode;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.teknek.deliverance.client.model.CreateChatCompletionRequest;
import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.nio.charset.StandardCharsets;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

class NanocodeDeliveranceRequestSizeTest {
    private static final ObjectMapper JSON = NanocodeDeliverance.clientMapper();

    @Test
    @SuppressWarnings({"rawtypes", "unchecked"})
    void fiveDefaultToolsProduceSmallChatRequest() throws Exception {
        NanocodeDeliverance agent = new NanocodeDeliverance(new NanocodeDeliverance.Config(
                "http://localhost:8085", "Llama-3.2-3B-Instruct-JQ4", null, 256, 2000, 0.0d, true, false, true, false));
        CreateChatCompletionRequest request = new CreateChatCompletionRequest()
                .model("Llama-3.2-3B-Instruct-JQ4")
                .maxTokens(256)
                .temperature(BigDecimal.ZERO)

                .messages((List) List.of(
                        NanocodeDeliverance.message("system", "You are a concise coding assistant. cwd: /tmp."),
                        NanocodeDeliverance.message("user", "hi")))
                .tools(agent.toolSchema())
                .parallelToolCalls(false);

        String json = JSON.writeValueAsString(request);
        int bytes = json.getBytes(StandardCharsets.UTF_8).length;
        System.out.println("nanocode request bytes=" + bytes);
        System.out.println(json);
        assertTrue(bytes < 20_000, "expected five-tool request to remain small, bytes=" + bytes);
    }

    @Test
    void systemPromptTellsModelToCallToolsInsteadOfDescribingJson() {
        String prompt = NanocodeDeliverance.systemPrompt("/tmp/work");

        assertTrue(prompt.contains("call the matching tool"));
        assertTrue(prompt.contains("do not describe the tool call or print JSON in prose"));
        assertTrue(prompt.contains("emit the tool call immediately"));
    }
}
