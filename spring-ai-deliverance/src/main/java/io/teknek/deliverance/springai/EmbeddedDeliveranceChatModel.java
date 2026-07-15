package io.teknek.deliverance.springai;

import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.CausalLanguageModel;
import io.teknek.deliverance.model.DoNothingGenerateEvent;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.MessageType;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.chat.prompt.Prompt;

import java.util.List;
import java.util.Objects;
import java.util.UUID;

public class EmbeddedDeliveranceChatModel implements ChatModel {
    private final CausalLanguageModel model;
    private final DeliveranceChatOptions defaultOptions;

    public EmbeddedDeliveranceChatModel(CausalLanguageModel model, DeliveranceChatOptions defaultOptions) {
        this.model = Objects.requireNonNull(model, "model");
        this.defaultOptions = Objects.requireNonNull(defaultOptions, "defaultOptions");
    }

    @Override
    public ChatResponse call(Prompt prompt) {
        DeliveranceChatOptions options = mergeOptions(prompt.getOptions());
        PromptContext promptContext = toPromptContext(prompt);
        Response response = model.generate(UUID.randomUUID(), promptContext,
                DeliveranceOptionsMapper.toGeneratorParameters(options), new DoNothingGenerateEvent());
        return new ChatResponse(List.of(new Generation(new AssistantMessage(response.responseText))));
    }

    @Override
    public ChatOptions getDefaultOptions() {
        return defaultOptions;
    }

    private PromptContext toPromptContext(Prompt prompt) {
        PromptSupport promptSupport = model.promptSupport()
                .orElseThrow(() -> new IllegalStateException("Deliverance model does not expose prompt support"));
        PromptSupport.Builder builder = promptSupport.builder();
        for (Message message : prompt.getInstructions()) {
            MessageType type = message.getMessageType();
            if (type == MessageType.SYSTEM) {
                builder.addSystemMessage(message.getText());
            } else if (type == MessageType.ASSISTANT) {
                builder.addAssistantMessage(message.getText());
            } else {
                builder.addUserMessage(message.getText());
            }
        }
        return builder.build();
    }

    private DeliveranceChatOptions mergeOptions(ChatOptions promptOptions) {
        if (promptOptions == null) {
            return defaultOptions;
        }
        if (promptOptions instanceof DeliveranceChatOptions deliveranceOptions) {
            return deliveranceOptions;
        }
        return DeliveranceChatOptions.builder()
                .model(promptOptions.getModel() == null ? defaultOptions.getModel() : promptOptions.getModel())
                .temperature(promptOptions.getTemperature() == null ? defaultOptions.getTemperature() : promptOptions.getTemperature())
                .maxTokens(promptOptions.getMaxTokens() == null ? defaultOptions.getMaxTokens() : promptOptions.getMaxTokens())
                .topP(promptOptions.getTopP() == null ? defaultOptions.getTopP() : promptOptions.getTopP())
                .topK(promptOptions.getTopK() == null ? defaultOptions.getTopK() : promptOptions.getTopK())
                .stopSequences(promptOptions.getStopSequences() == null ? defaultOptions.getStopSequences() : promptOptions.getStopSequences())
                .build();
    }
}
