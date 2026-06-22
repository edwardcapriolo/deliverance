package io.teknek.deliverance.safetensors.prompt.hf;

import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

/**
 * Directly named ports of Hugging Face Mistral common prompt-template tests.
 *
 * <p>Source: /ai-code/transformers/tests/test_tokenization_mistral_common.py</p>
 */
public class HfMistralPromptTemplateCommonPortTest {

    @Disabled("HF compares against mistral_common reference tokenizer; Deliverance has local Mistral render regressions separately")
    @Test public void testApplyChatTemplateBasic() {}

    @Disabled("Deliverance PromptSupport does not currently expose continue_final_message")
    @Test public void testApplyChatTemplateContinueFinalMessage() {}

    @Disabled("Deliverance PromptSupport does not currently expose tokenize=True apply_chat_template API")
    @Test public void testApplyChatTemplateWithAddGenerationPrompt() {}

    @Disabled("HF compares exact Mistral tool rendering against mistral_common reference tokenizer; local smoke coverage is separate")
    @Test public void testApplyChatTemplateWithTools() {}

    @Disabled("Deliverance does not currently implement multimodal image prompt processing")
    @Test public void testApplyChatTemplateWithImage() {}

    @Disabled("Deliverance does not currently implement multimodal audio prompt processing")
    @Test public void testApplyChatTemplateWithAudio() {}

    @Disabled("Deliverance PromptSupport does not currently expose tokenize=True truncation API")
    @Test public void testApplyChatTemplateWithTruncation() {}

    @Disabled("Deliverance PromptSupport currently renders one conversation per builder and has no batched apply_chat_template API")
    @Test public void testBatchApplyChatTemplate() {}

    @Disabled("Deliverance does not currently implement batched multimodal image prompt processing")
    @Test public void testBatchApplyChatTemplateImages() {}

    @Disabled("Deliverance PromptSupport does not currently expose batched continue_final_message")
    @Test public void testBatchApplyChatTemplateWithContinueFinalMessage() {}

    @Disabled("Deliverance PromptSupport does not currently expose batched tokenize=True apply_chat_template API")
    @Test public void testBatchApplyChatTemplateWithAddGenerationPrompt() {}

    @Disabled("Deliverance PromptSupport does not currently expose batched tokenize=True truncation API")
    @Test public void testBatchApplyChatTemplateWithTruncation() {}

    @Disabled("Deliverance PromptSupport does not currently expose batched tokenize=True padding API")
    @Test public void testBatchApplyChatTemplateWithPadding() {}

    @Disabled("Deliverance PromptSupport does not currently expose batched tokenize=True padding/truncation API")
    @Test public void testBatchApplyChatTemplateWithPaddingAndTruncation() {}

    @Disabled("Deliverance PromptSupport does not currently expose return_tensors in apply_chat_template")
    @Test public void testBatchApplyChatTemplateReturnTensors() {}

    @Disabled("Deliverance PromptSupport does not currently expose return_dict in apply_chat_template")
    @Test public void testBatchApplyChatTemplateReturnDict() {}
}
