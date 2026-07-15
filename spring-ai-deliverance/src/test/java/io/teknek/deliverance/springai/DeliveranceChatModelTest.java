package io.teknek.deliverance.springai;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.teknek.deliverance.client.model.CreateChatCompletionRequest;
import io.teknek.deliverance.client.model.CreateChatCompletionResponse;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.prompt.Prompt;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DeliveranceChatModelTest {

    @Test
    void mapsPromptAndOptionsToChatCompletionRequest() {
        DeliveranceChatOptions options = DeliveranceChatOptions.builder()
                .model("edwardcapriolo/Qwen3-4B-JQ4")
                .temperature(0.0)
                .maxTokens(64)
                .topP(0.95)
                .topK(64)
                .seed(42)
                .logprobs(true)
                .topLogprobs(5)
                .guidedRegex("TICKET-[0-9]{4}")
                .build();
        DeliveranceChatModel model = new DeliveranceChatModel(new NoopDeliveranceApi(), new ObjectMapper(), options);

        CreateChatCompletionRequest request = model.toRequest(new Prompt(List.of(
                new SystemMessage("You are concise."),
                new UserMessage("Create a ticket id."))), false);

        assertEquals("edwardcapriolo/Qwen3-4B-JQ4", request.getModel());
        assertEquals(false, request.getStream());
        assertEquals("system", request.getMessages().get(0).getRole());
        assertEquals("user", request.getMessages().get(1).getRole());
        assertEquals(0.0, request.getTemperature().doubleValue());
        assertEquals(64, request.getMaxTokens());
        assertEquals(0.95, request.getTopP().doubleValue());
        assertEquals(64, request.getTopK().intValue());
        assertEquals(42, request.getSeed());
        assertEquals(true, request.getLogprobs());
        assertEquals(5, request.getTopLogprobs());
        assertEquals("TICKET-[0-9]{4}", request.getGuidedRegex());
    }

    @Test
    void mapsGuidedJsonAsObject() {
        DeliveranceChatOptions options = DeliveranceChatOptions.builder()
                .model("edwardcapriolo/Qwen3-4B-JQ4")
                .guidedJson("""
                        {"type":"object","properties":{"foo":{"type":"integer"}},"required":["foo"]}
                        """)
                .build();
        DeliveranceChatModel model = new DeliveranceChatModel(new NoopDeliveranceApi(), new ObjectMapper(), options);

        CreateChatCompletionRequest request = model.toRequest(new Prompt("Return JSON."), false);

        assertTrue(request.getGuidedJson().containsKey("properties"));
        assertEquals("integer", ((java.util.Map<?, ?>) ((java.util.Map<?, ?>) request.getGuidedJson()
                .get("properties")).get("foo")).get("type"));
    }

    private static final class NoopDeliveranceApi implements DeliveranceApi {
        @Override
        public CreateChatCompletionResponse createChatCompletion(CreateChatCompletionRequest request) {
            return null;
        }
    }
}
