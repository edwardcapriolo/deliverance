package io.teknek.deliverance.safetensors.prompt.hf;

import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Render-only port of Hugging Face prompt-template common behavior.
 *
 * <p>Primary sources:</p>
 * <ul>
 *     <li>/ai-code/transformers/tests/test_tokenization_common.py TokenizerTesterMixin.test_chat_template</li>
 *     <li>/ai-code/transformers/tests/test_processing_common.py test_chat_template_jinja_kwargs</li>
 *     <li>/ai-code/transformers/tests/test_tokenization_mistral_common.py role-validation template cases</li>
 * </ul>
 */
public interface HfPromptTemplateContract {

    String hfDummyTemplate();

    String hfDummyExpectedOutput();

    default PromptSupport promptSupport(String defaultTemplate) {
        return new PromptSupport(Map.of("default", defaultTemplate), "", true);
    }

    default PromptSupport.Builder hfDummyConversation(PromptSupport promptSupport) {
        return promptSupport.builder()
                .addSystemMessage("system message")
                .addUserMessage("user message")
                .addAssistantMessage("assistant message");
    }

    @Test
    default void chatTemplateCanBePassedAsOverride() {
        PromptContext output = hfDummyConversation(promptSupport("unused"))
                .useChatTemplate(hfDummyTemplate())
                .build();

        assertEquals(hfDummyExpectedOutput(), output.getPrompt());
    }

    @Test
    default void chatTemplateAttributeIsUsedWhenNoOverrideIsPassed() {
        PromptContext output = hfDummyConversation(promptSupport(hfDummyTemplate()))
                .build();

        assertEquals(hfDummyExpectedOutput(), output.getPrompt());
    }

    @Test
    default void addGenerationPromptIsAvailableToJinjaTemplate() {
        String template = "{% for message in messages %}{{ message['role'] + ':' + message['content'] + '\n' }}{% endfor %}"
                + "{% if add_generation_prompt %}assistant:{% endif %}";

        PromptContext withGenerationPrompt = promptSupport(template).builder()
                .addUserMessage("hello")
                .addGenerationPrompt(true)
                .build();
        PromptContext withoutGenerationPrompt = promptSupport(template).builder()
                .addUserMessage("hello")
                .addGenerationPrompt(false)
                .build();

        assertEquals("user:hello\nassistant:", withGenerationPrompt.getPrompt());
        assertEquals("user:hello\n", withoutGenerationPrompt.getPrompt());
    }

    @Test
    default void customJinjaKwargsAreAvailableToTemplate() {
        String template = "{% for message in messages %}"
                + "{% if add_system_prompt %}{{ 'You are a helpful assistant.' }}{% endif %}"
                + "{{ '<|special_start|>' + message['role'] + '\n' + message['content'] + '<|special_end|>' + '\n' }}"
                + "{% endfor %}";

        PromptContext output = promptSupport(template).builder()
                .addTemplateArg("add_system_prompt", true)
                .addUserMessage("Which of these animals is making the sound?")
                .addAssistantMessage("It is a cow.")
                .build();

        assertEquals("You are a helpful assistant.<|special_start|>user\n"
                + "Which of these animals is making the sound?<|special_end|>\n"
                + "You are a helpful assistant.<|special_start|>assistant\n"
                + "It is a cow.<|special_end|>\n", output.getPrompt());
    }

    @Test
    default void raiseExceptionFunctionFailsPromptRendering() {
        String template = "{% for message in messages %}"
                + "{% if (message['role'] == 'assistant') %}{{ raise_exception('bad role') }}{% endif %}"
                + "{{ message['content'] }}"
                + "{% endfor %}";

        assertThrows(RuntimeException.class, () -> promptSupport(template).builder()
                .addUserMessage("hello")
                .addAssistantMessage("bad")
                .build());
    }
}
