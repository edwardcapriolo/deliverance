package io.teknek.deliverance.safetensors.prompt.local;

import io.teknek.deliverance.safetensors.prompt.Function;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import io.teknek.deliverance.safetensors.prompt.Tool;
import io.teknek.deliverance.safetensors.prompt.ToolCall;
import io.teknek.deliverance.safetensors.prompt.ToolResult;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class LocalMistralPromptTemplateTest implements LocalPromptTemplateFamilyContract {
    @Override
    public PromptSupport familyPromptSupport() {
        return LocalPromptTemplateFixtures.promptSupport("tjake_Mistral-7B-Instruct-v0.3-JQ4");
    }

    @Override public String expectedBasicUserPrompt() { return "<s>[INST] Hi![/INST]"; }
    @Override public String expectedSystemUserPrompt() { return "<s>[INST] You are helpful.\n\nHi![/INST]"; }
    @Override public String expectedMultiTurnPrompt() { return "<s>[INST] Hi![/INST] Hello!</s>[INST] What next?[/INST]"; }

    @Test
    public void rendersToolsAndToolResultsSmoke() {
        Tool tool = Tool.from(Function.builder().name("get_weather").description("Get weather.")
                .addParameter("location", "string", "City", true).build());
        assertTrue(familyPromptSupport().builder().addToolItem(tool).addUserMessage("Weather?").build().getPrompt().contains("[AVAILABLE_TOOLS]"));

        ToolCall call = new ToolCall("get_weather", "call12345", Map.of("location", "Paris"));
        PromptContext result = familyPromptSupport().builder()
                .addUserMessage("Weather?")
                .addToolCall(call)
                .addToolResult(ToolResult.from("get_weather", "call12345", "22"))
                .build();
        assertTrue(result.getPrompt().contains("get_weather"), result.getPrompt());
        assertTrue(result.getPrompt().contains("22"), result.getPrompt());
    }
}
