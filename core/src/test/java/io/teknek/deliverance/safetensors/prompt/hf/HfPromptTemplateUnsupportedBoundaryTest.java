package io.teknek.deliverance.safetensors.prompt.hf;

import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Explicit boundaries for Hugging Face apply_chat_template features Deliverance does not currently expose.
 */
public class HfPromptTemplateUnsupportedBoundaryTest {

    @Test
    public void continueFinalMessageApiIsNotCurrentlyExposed() {
        boolean hasApi = Arrays.stream(PromptSupport.Builder.class.getMethods())
                .map(Method::getName)
                .anyMatch(name -> name.equals("continueFinalMessage") || name.equals("continue_final_message"));

        assertFalse(hasApi, "If this API is added, replace this boundary test with HF continue_final_message contract tests");
    }

    @Test
    public void generationBlocksAreNotCurrentlySupportedByJinjavaRenderer() {
        PromptSupport support = new PromptSupport(java.util.Map.of("default", "{% generation %}hello{% endgeneration %}"), "", true);

        assertThrows(RuntimeException.class, () -> support.builder().addUserMessage("hi").build());
    }

    @Test
    public void batchedApplyChatTemplateApiIsNotCurrentlyExposed() {
        boolean hasBatchApi = Arrays.stream(PromptSupport.class.getMethods())
                .anyMatch(method -> method.getName().equals("applyChatTemplate")
                        && method.getParameterCount() > 0
                        && List.class.isAssignableFrom(method.getParameterTypes()[0]));

        assertFalse(hasBatchApi, "If batched prompt rendering is added, replace this with HF batch apply_chat_template contract tests");
    }

    @Test
    public void renderAndTokenizeReturnDictApiIsNotCurrentlyExposed() {
        boolean hasRenderTokenizeApi = Arrays.stream(PromptSupport.class.getMethods())
                .anyMatch(method -> method.getName().equals("applyChatTemplate") || method.getName().equals("renderAndTokenize"));

        assertFalse(hasRenderTokenizeApi, "If render+tokenize return-dict API is added, replace this with HF tokenize=True return_dict contract tests");
    }

    @Test
    public void multimodalProcessorApiIsNotCurrentlyExposed() {
        boolean hasMultimodalApi = Arrays.stream(PromptSupport.Builder.class.getMethods())
                .anyMatch(method -> method.getName().equals("addImage")
                        || method.getName().equals("addAudio")
                        || method.getName().equals("addVideo")
                        || method.getName().equals("addMultimodalMessage"));

        assertFalse(hasMultimodalApi, "If multimodal prompt processing is added, replace this with HF processor multimodal contract tests");
    }
}
