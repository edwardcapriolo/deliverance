package io.teknek.deliverance.safetensors.prompt.local;

import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import org.junit.jupiter.api.Assertions;

public class LocalGemma2PromptTemplateTest implements LocalPromptTemplateFamilyContract {
    @Override public PromptSupport familyPromptSupport() { return LocalPromptTemplateFixtures.promptSupport("tjake_gemma-2-2b-it-JQ4"); }
    @Override public String expectedBasicUserPrompt() { return "<bos><start_of_turn>user\nHi!<end_of_turn>\n<start_of_turn>model\n"; }
    @Override public String expectedSystemUserPrompt() { return "__UNSUPPORTED__"; }
    @Override public String expectedMultiTurnPrompt() { return "<bos><start_of_turn>user\nHi!<end_of_turn>\n<start_of_turn>model\nHello!<end_of_turn>\n<start_of_turn>user\nWhat next?<end_of_turn>\n<start_of_turn>model\n"; }

    @Override
    public void rendersSystemAndUserPrompt() {
        Assertions.assertThrows(RuntimeException.class, () -> familyPromptSupport().builder().addSystemMessage("You are helpful.").addUserMessage("Hi!").build());
    }
}
