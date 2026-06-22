package io.teknek.deliverance.safetensors.prompt.local;

import io.teknek.deliverance.safetensors.prompt.PromptSupport;

public class LocalTinyLlamaPromptTemplateTest implements LocalPromptTemplateFamilyContract {
    @Override public PromptSupport familyPromptSupport() { return LocalPromptTemplateFixtures.promptSupport("tjake_TinyLlama-1.1B-Chat-v1.0-Jlama-Q4"); }
    @Override public String expectedBasicUserPrompt() { return "<|user|>\nHi!</s>\n<|assistant|>\n"; }
    @Override public String expectedSystemUserPrompt() { return "<|system|>\nYou are helpful.</s>\n<|user|>\nHi!</s>\n<|assistant|>\n"; }
    @Override public String expectedMultiTurnPrompt() { return "<|user|>\nHi!</s>\n<|assistant|>\nHello!</s>\n<|user|>\nWhat next?</s>\n<|assistant|>\n"; }
}
