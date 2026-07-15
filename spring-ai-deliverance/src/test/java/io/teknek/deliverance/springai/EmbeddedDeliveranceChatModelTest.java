package io.teknek.deliverance.springai;

import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.CausalLanguageModel;
import io.teknek.deliverance.model.GenerateEvent;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;

import java.util.List;
import java.util.Optional;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class EmbeddedDeliveranceChatModelTest {

    @Test
    void mapsPromptThroughPromptSupportAndGeneratorParameters() {
        CausalLanguageModel causalLanguageModel = mock(CausalLanguageModel.class);
        PromptSupport promptSupport = new PromptSupport(java.util.Map.of("default", "{{ messages }}"), "", "", false);
        when(causalLanguageModel.promptSupport()).thenReturn(Optional.of(promptSupport));
        when(causalLanguageModel.generate(any(UUID.class), any(PromptContext.class), any(GeneratorParameters.class), any(GenerateEvent.class)))
                .thenReturn(new Response("embedded response", "embedded response", null, 0, List.of(), 0, 0, List.of()));

        EmbeddedDeliveranceChatModel model = new EmbeddedDeliveranceChatModel(causalLanguageModel,
                DeliveranceChatOptions.builder().model("owner/model").temperature(0.0).maxTokens(32).build());

        ChatResponse response = model.call(new Prompt(List.of(new SystemMessage("Be brief"), new UserMessage("Hi"))));

        assertEquals("embedded response", response.getResult().getOutput().getText());
        ArgumentCaptor<GeneratorParameters> parameters = ArgumentCaptor.forClass(GeneratorParameters.class);
        org.mockito.Mockito.verify(causalLanguageModel).generate(any(UUID.class), any(PromptContext.class), parameters.capture(), any(GenerateEvent.class));
        assertEquals(0.0f, parameters.getValue().temperature.orElseThrow());
        assertEquals(32, parameters.getValue().maxTokens.orElseThrow());
    }
}
