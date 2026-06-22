package io.teknek.deliverance.safetensors.prompt.local;

import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** Regression checks for Deliverance's local cached prompt templates. These are not Hugging Face golden assertions. */
public interface LocalPromptTemplateFamilyContract {

    PromptSupport familyPromptSupport();

    String expectedBasicUserPrompt();

    String expectedSystemUserPrompt();

    String expectedMultiTurnPrompt();

    @Test
    default void rendersBasicUserPrompt() {
        PromptContext prompt = familyPromptSupport().builder()
                .addUserMessage("Hi!")
                .build();

        assertEquals(expectedBasicUserPrompt(), prompt.getPrompt());
    }

    @Test
    default void rendersSystemAndUserPrompt() {
        PromptContext prompt = familyPromptSupport().builder()
                .addSystemMessage("You are helpful.")
                .addUserMessage("Hi!")
                .build();

        assertEquals(expectedSystemUserPrompt(), prompt.getPrompt());
    }

    @Test
    default void rendersMultiTurnPromptWithAssistantHistory() {
        PromptContext prompt = familyPromptSupport().builder()
                .addUserMessage("Hi!")
                .addAssistantMessage("Hello!")
                .addUserMessage("What next?")
                .build();

        assertEquals(expectedMultiTurnPrompt(), prompt.getPrompt());
    }
}
