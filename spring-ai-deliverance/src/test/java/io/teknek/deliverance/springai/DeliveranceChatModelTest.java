package io.teknek.deliverance.springai;

import io.teknek.deliverance.client.spring.model.CreateChatCompletionRequest;
import io.teknek.deliverance.client.spring.model.CreateChatCompletionResponse;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.ToolResponseMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.prompt.Prompt;
import tools.jackson.databind.ObjectMapper;

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
                .xtcThreshold(0.5)
                .xtcProbability(0.1)
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
        assertEquals(0.5, request.getXtcThreshold().doubleValue());
        assertEquals(0.1, request.getXtcProbability().doubleValue());
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

    @Test
    void runtimeDeliveranceOptionsMergeWithDefaults() {
        DeliveranceChatOptions defaults = DeliveranceChatOptions.builder()
                .model("default-model")
                .temperature(0.7)
                .maxTokens(128)
                .topP(0.9)
                .topK(40)
                .seed(42)
                .xtcThreshold(0.5)
                .guidedRegex("DEFAULT-[0-9]+")
                .build();
        DeliveranceChatOptions runtime = DeliveranceChatOptions.builder()
                .temperature(0.0)
                .maxTokens(64)
                .build();
        DeliveranceChatModel model = new DeliveranceChatModel(new NoopDeliveranceApi(), new ObjectMapper(), defaults);

        CreateChatCompletionRequest request = model.toRequest(new Prompt("hello", runtime), false);

        assertEquals("default-model", request.getModel());
        assertEquals(0.0, request.getTemperature().doubleValue());
        assertEquals(64, request.getMaxTokens());
        assertEquals(0.9, request.getTopP().doubleValue());
        assertEquals(40, request.getTopK().intValue());
        assertEquals(42, request.getSeed());
        assertEquals(0.5, request.getXtcThreshold().doubleValue());
        assertEquals("DEFAULT-[0-9]+", request.getGuidedRegex());
    }

    @Test
    void serializesRequestWithSpringGeneratedMapper() throws Exception {
        DeliveranceChatOptions options = DeliveranceChatOptions.builder()
                .model("edwardcapriolo/Qwen3-4B-JQ4")
                .temperature(0.0)
                .maxTokens(64)
                .topK(64)
                .xtcThreshold(0.5)
                .guidedJson("""
                        {"type":"object","properties":{"foo":{"type":"integer"}},"required":["foo"]}
                        """)
                .build();
        DeliveranceChatModel model = new DeliveranceChatModel(new NoopDeliveranceApi(), new ObjectMapper(), options);

        CreateChatCompletionRequest request = model.toRequest(new Prompt(List.of(
                new SystemMessage("You are concise."),
                new UserMessage("Return foo as JSON."))), false);

        String json = DeliveranceApi.jsonMapper().writeValueAsString(request);

        assertTrue(json.contains("\"model\":\"edwardcapriolo/Qwen3-4B-JQ4\""));
        assertTrue(json.contains("\"role\":\"system\""), json);
        assertTrue(json.contains("\"role\":\"user\""), json);
        assertTrue(json.contains("\"top_k\":64"), json);
        assertTrue(json.contains("\"xtc_threshold\":0.5"), json);
        assertTrue(json.contains("\"guided_json\""), json);
        assertTrue(json.contains("\"foo\""), json);
        assertTrue(!json.contains("\"stop\":null"), json);
    }

    @Test
    void mapsToolCallConversationHistoryToRequest() {
        DeliveranceChatModel model = new DeliveranceChatModel(new NoopDeliveranceApi(), new ObjectMapper(),
                DeliveranceChatOptions.builder().model("test-model").build());
        AssistantMessage assistantMessage = AssistantMessage.builder()
                .content("")
                .toolCalls(List.of(new AssistantMessage.ToolCall("call_1", "function", "getCurrentWeather",
                        "{\"location\":\"San Francisco, CA\",\"unit\":\"C\"}")))
                .build();
        ToolResponseMessage toolResponseMessage = ToolResponseMessage.builder()
                .responses(List.of(new ToolResponseMessage.ToolResponse("call_1", "getCurrentWeather",
                        "{\"temp\":30,\"unit\":\"C\"}")))
                .build();

        CreateChatCompletionRequest request = model.toRequest(new Prompt(List.of(
                new UserMessage("What are the weather conditions in San Francisco?"),
                assistantMessage,
                toolResponseMessage)), false);

        assertEquals("user", request.getMessages().get(0).getRole());
        assertEquals("assistant", request.getMessages().get(1).getRole());
        assertEquals(1, request.getMessages().get(1).getToolCalls().size());
        assertEquals("tool", request.getMessages().get(2).getRole());
        assertEquals("call_1", request.getMessages().get(2).getToolCallId());
        assertTrue(request.getMessages().get(2).getContent().contains("30"));
    }

    private static final class NoopDeliveranceApi implements DeliveranceApi {
        @Override
        public CreateChatCompletionResponse createChatCompletion(CreateChatCompletionRequest request) {
            return null;
        }
    }
}
