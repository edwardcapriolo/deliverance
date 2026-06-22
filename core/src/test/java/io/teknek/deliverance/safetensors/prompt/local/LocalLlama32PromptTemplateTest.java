package io.teknek.deliverance.safetensors.prompt.local;

import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class LocalLlama32PromptTemplateTest implements LocalPromptTemplateFamilyContract {
    @Override public PromptSupport familyPromptSupport() { return LocalPromptTemplateFixtures.promptSupport("tjake_Llama-3.2-1B-Instruct-JQ4"); }
    @Override public String expectedBasicUserPrompt() { return "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHi!<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"; }
    @Override public String expectedSystemUserPrompt() { return "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nYou are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHi!<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"; }
    @Override public String expectedMultiTurnPrompt() { return "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHi!<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nHello!<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat next?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"; }

    @Test public void rendersCustomDateTemplateArg() { Assertions.assertTrue(familyPromptSupport().builder().addTemplateArg("date_string", "01 Jan 2026").addUserMessage("Hi!").build().getPrompt().contains("Today Date: 01 Jan 2026")); }
}
