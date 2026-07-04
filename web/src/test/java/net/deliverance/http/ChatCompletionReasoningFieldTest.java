package net.deliverance.http;

import com.fasterxml.jackson.databind.JsonNode;
import io.teknek.deliverance.JsonUtils;
import io.teknek.deliverance.model.ChatCompletionResponseMessage;
import io.teknek.deliverance.model.ReasoningFieldNames;
import org.junit.jupiter.api.Test;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ChatCompletionReasoningFieldTest {
    @Test
    void openAiReasoningModeUsesReasoningContentOnly() {
        ChatCompletionController controller = new ChatCompletionController(Optional.empty(), false, "off", "openai", false);
        ChatCompletionResponseMessage message = new ChatCompletionResponseMessage()
                .role(ChatCompletionResponseMessage.RoleEnum.ASSISTANT)
                .content("Albany");

        controller.applyReasoning(message, "thinking");
        JsonNode json = JsonUtils.om.valueToTree(message);

        assertEquals("thinking", json.path(ReasoningFieldNames.OPENAI).asText());
        assertTrue(!json.has(ReasoningFieldNames.VLLM) || json.get(ReasoningFieldNames.VLLM).isNull());
    }

    @Test
    void vllmReasoningModeUsesReasoningOnly() {
        ChatCompletionController controller = new ChatCompletionController(Optional.empty(), false, "off", "vllm", false);
        ChatCompletionResponseMessage message = new ChatCompletionResponseMessage()
                .role(ChatCompletionResponseMessage.RoleEnum.ASSISTANT)
                .content("Albany");

        controller.applyReasoning(message, "thinking");
        JsonNode json = JsonUtils.om.valueToTree(message);

        assertEquals("thinking", json.path(ReasoningFieldNames.VLLM).asText());
        assertTrue(!json.has(ReasoningFieldNames.OPENAI) || json.get(ReasoningFieldNames.OPENAI).isNull());
    }
}
