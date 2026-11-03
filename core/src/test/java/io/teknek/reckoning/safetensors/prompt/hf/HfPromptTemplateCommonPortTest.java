package io.teknek.reckoning.safetensors.prompt.hf;

import io.teknek.reckoning.safetensors.prompt.PromptContext;
import io.teknek.reckoning.safetensors.prompt.PromptSupport;
import org.junit.jupiter.api.Disabled;
import io.teknek.reckoning.model.ReasoningFieldNames;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Directly named ports of Hugging Face common chat-template tests.
 *
 * <p>Source: /ai-code/transformers/tests/test_tokenization_common.py</p>
 */
public class HfPromptTemplateCommonPortTest {

    @Test
    public void testChatTemplate() {
        String dummyTemplate = "{% for message in messages %}{{message['role'] + message['content']}}{% endfor %}";
        PromptContext output = promptSupport(dummyTemplate).builder()
                .addSystemMessage("system message")
                .addUserMessage("user message")
                .addAssistantMessage("assistant message")
                .build();

        assertEquals("systemsystem messageuseruser messageassistantassistant message", output.getPrompt());
    }

    @Disabled("Reckoning PromptSupport has no save_pretrained/from_pretrained tokenizer-template persistence API")
    @Test
    public void testChatTemplateSaveLoading() {
    }

    @Disabled("Reckoning PromptSupport currently renders one conversation per builder and has no batched apply_chat_template API")
    @Test
    public void testChatTemplateBatched() {
    }

    @Disabled("Jinjava in Reckoning does not currently expose Python Jinja loopcontrols extension")
    @Test
    public void testJinjaLoopcontrols() {
    }

    @Disabled("Reckoning PromptSupport does not currently expose strftime_now in template context")
    @Test
    public void testJinjaStrftime() {
    }

    @Disabled("Reckoning does not currently implement {% generation %} blocks or assistant token masks")
    @Test
    public void testChatTemplateReturnAssistantTokensMask() {
    }

    @Disabled("Reckoning does not currently implement {% generation %} blocks or assistant token masks")
    @Test
    public void testChatTemplateReturnAssistantTokensMaskTruncated() {
    }

    @Disabled("Reckoning PromptSupport does not currently expose continue_final_message")
    @Test
    public void testContinueFinalMessage() {
    }

    @Disabled("Reckoning PromptSupport does not currently expose continue_final_message")
    @Test
    public void testContinueFinalMessageWithTrim() {
    }

    @Disabled("Reckoning PromptSupport does not currently expose continue_final_message")
    @Test
    public void testContinueFinalMessageWithDecoyEarlierMessage() {
    }

    @Disabled("Reckoning PromptSupport does not currently expose continue_final_message='" + ReasoningFieldNames.OPENAI + "'")
    @Test
    public void testContinueFinalMessageStringAndReasoning() {
    }

    @Disabled("Reckoning PromptSupport currently stores a single default template map and does not expose HF chat_template dict selection")
    @Test
    public void testChatTemplateDict() {
    }

    @Disabled("Reckoning PromptSupport has no save_pretrained/from_pretrained tokenizer-template persistence API")
    @Test
    public void testChatTemplateDictSaving() {
    }

    @Disabled("Reckoning PromptSupport has no save_pretrained/from_pretrained tokenizer-template persistence API")
    @Test
    public void testChatTemplateDictSavingRejectsPathTraversal() {
    }

    @Disabled("Reckoning PromptSupport has no tokenizer save/load file-priority API")
    @Test
    public void testChatTemplateFilePriority() {
    }

    @Test
    public void raiseExceptionFunctionIsAvailableLikeHfTemplatesExpect() {
        PromptSupport support = promptSupport("{{ raise_exception('bad role') }}");

        assertThrows(RuntimeException.class, () -> support.builder().addUserMessage("hi").build());
    }

    private static PromptSupport promptSupport(String template) {
        return new PromptSupport(Map.of("default", template), "", "", true);
    }
}
