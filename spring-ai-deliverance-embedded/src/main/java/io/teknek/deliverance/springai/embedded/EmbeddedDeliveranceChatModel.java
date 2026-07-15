package io.teknek.deliverance.springai.embedded;

import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.CausalLanguageModel;
import io.teknek.deliverance.model.DoNothingGenerateEvent;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import io.teknek.deliverance.springai.DeliveranceChatOptions;

import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.chat.prompt.Prompt;

import java.util.List;
import java.util.Map;
import java.util.UUID;

public class EmbeddedDeliveranceChatModel implements ChatModel {
    private final CausalLanguageModel model;
    private final DeliveranceChatOptions options;

    public EmbeddedDeliveranceChatModel(CausalLanguageModel model, DeliveranceChatOptions options) {
        this.model = model;
        this.options = options;
    }

    @Override
    public ChatResponse call(Prompt prompt) {
        PromptContext promptContext = toPromptContext(prompt);
        Response response = model.generate(UUID.randomUUID(), promptContext,
                EmbeddedDeliveranceOptionsMapper.toGeneratorParameters(mergeOptions(prompt.getOptions())),
                new DoNothingGenerateEvent());
        return new ChatResponse(List.of(new Generation(new AssistantMessage(response.responseText))));
    }

    @Override
    public ChatOptions getOptions() {
        return options;
    }

    private PromptContext toPromptContext(Prompt prompt) {
        PromptSupport promptSupport = model.promptSupport()
                .orElseThrow(() -> new IllegalStateException("Embedded Deliverance model has no PromptSupport"));
        PromptSupport.Builder builder = promptSupport.builder();
        prompt.getInstructions().forEach(message -> {
            switch (message.getMessageType()) {
                case SYSTEM -> builder.addSystemMessage(message.getText());
                case ASSISTANT -> builder.addAssistantMessage(message.getText());
                case TOOL -> builder.addToolResult(io.teknek.deliverance.safetensors.prompt.ToolResult.from("tool",
                        "tool", message.getText()));
                default -> builder.addUserMessage(message.getText());
            }
        });
        return builder.build();
    }

    private DeliveranceChatOptions mergeOptions(ChatOptions promptOptions) {
        if (promptOptions == null) {
            return options;
        }
        if (promptOptions instanceof DeliveranceChatOptions deliveranceOptions) {
            return deliveranceOptions;
        }
        return DeliveranceChatOptions.builder()
                .model(promptOptions.getModel() == null ? options.getModel() : promptOptions.getModel())
                .temperature(promptOptions.getTemperature() == null ? options.getTemperature() : promptOptions.getTemperature())
                .maxTokens(promptOptions.getMaxTokens() == null ? options.getMaxTokens() : promptOptions.getMaxTokens())
                .topP(promptOptions.getTopP() == null ? options.getTopP() : promptOptions.getTopP())
                .topK(promptOptions.getTopK() == null ? options.getTopK() : promptOptions.getTopK())
                .stopSequences(promptOptions.getStopSequences() == null ? options.getStopSequences() : promptOptions.getStopSequences())
                .build();
    }
}
