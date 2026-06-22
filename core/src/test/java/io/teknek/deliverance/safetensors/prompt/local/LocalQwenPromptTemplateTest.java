package io.teknek.deliverance.safetensors.prompt.local;

import io.teknek.deliverance.safetensors.prompt.Function;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import io.teknek.deliverance.safetensors.prompt.Tool;
import io.teknek.deliverance.safetensors.prompt.ToolCall;
import io.teknek.deliverance.safetensors.prompt.ToolResult;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class LocalQwenPromptTemplateTest implements LocalPromptTemplateFamilyContract {
    @Override
    public PromptSupport familyPromptSupport() {
        return LocalPromptTemplateFixtures.promptSupport("Qwen_Qwen2.5-7B-Instruct");
    }

    @Override
    public String expectedBasicUserPrompt() {
        return """
                <|im_start|>system
                You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
                <|im_start|>user
                Hi!<|im_end|>
                <|im_start|>assistant
                """;
    }

    @Override
    public String expectedSystemUserPrompt() {
        return """
                <|im_start|>system
                You are helpful.<|im_end|>
                <|im_start|>user
                Hi!<|im_end|>
                <|im_start|>assistant
                """;
    }

    @Override
    public String expectedMultiTurnPrompt() {
        return """
                <|im_start|>system
                You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
                <|im_start|>user
                Hi!<|im_end|>
                <|im_start|>assistant
                Hello!<|im_end|>
                <|im_start|>user
                What next?<|im_end|>
                <|im_start|>assistant
                """;
    }

    @Test
    public void rendersNestedToolSchemaAndToolResultSmoke() {
        Tool tool = Tool.from(Function.builder()
                .name("search_hotels")
                .description("Search hotels.")
                .addParameter("query", Map.of(
                        "type", "object",
                        "properties", Map.of("city", Map.of("type", "string"), "guests", Map.of("type", "integer"))),
                        true)
                .build());
        PromptContext declaration = familyPromptSupport().builder().addToolItem(tool).addUserMessage("Find a hotel.").build();
        assertTrue(declaration.getPrompt().contains("search_hotels"), declaration.getPrompt());

        ToolCall call = new ToolCall("search_hotels", "call_1", Map.of("query", Map.of("city", "Paris")));
        PromptContext result = familyPromptSupport().builder()
                .addUserMessage("Find a hotel.")
                .addToolCall(call)
                .addToolResult(ToolResult.from("search_hotels", "call_1", "done"))
                .build();
        assertTrue(result.getPrompt().contains("<tool_response>"), result.getPrompt());
        assertTrue(result.getPrompt().contains("done"), result.getPrompt());
    }
}
