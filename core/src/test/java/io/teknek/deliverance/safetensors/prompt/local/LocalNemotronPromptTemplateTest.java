package io.teknek.deliverance.safetensors.prompt.local;

import io.teknek.deliverance.safetensors.prompt.PromptSupport;

public class LocalNemotronPromptTemplateTest implements LocalPromptTemplateFamilyContract {
    @Override
    public PromptSupport familyPromptSupport() {
        return LocalPromptTemplateFixtures.promptSupport("nvidia_Nemotron-Labs-Diffusion-3B-Base");
    }

    @Override
    public String expectedBasicUserPrompt() {
        return """
                <|im_start|>system
                <|im_end|>
                <|im_start|>user
                Hi!<|im_end|>
                <|im_start|>assistant
                <think></think>""";
    }

    @Override
    public String expectedSystemUserPrompt() {
        return """
                <|im_start|>system
                You are helpful.<|im_end|>
                <|im_start|>user
                Hi!<|im_end|>
                <|im_start|>assistant
                <think></think>""";
    }

    @Override
    public String expectedMultiTurnPrompt() {
        return """
                <|im_start|>system
                <|im_end|>
                <|im_start|>user
                Hi!<|im_end|>
                <|im_start|>assistant
                <think></think>Hello!<|im_end|>
                <|im_start|>user
                What next?<|im_end|>
                <|im_start|>assistant
                <think></think>""";
    }
}
