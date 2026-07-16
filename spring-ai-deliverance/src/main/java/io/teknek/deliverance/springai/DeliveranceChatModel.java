package io.teknek.deliverance.springai;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.teknek.deliverance.client.model.ChatCompletionRequestMessage;
import io.teknek.deliverance.client.model.CreateChatCompletionRequest;
import io.teknek.deliverance.client.model.CreateChatCompletionResponse;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.MessageType;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.chat.prompt.Prompt;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

public class DeliveranceChatModel implements ChatModel {
    private final DeliveranceApi deliveranceApi;
    private final ObjectMapper objectMapper;
    private final DeliveranceChatOptions defaultOptions;

    public DeliveranceChatModel(DeliveranceApi deliveranceApi, ObjectMapper objectMapper, DeliveranceChatOptions defaultOptions) {
        this.deliveranceApi = Objects.requireNonNull(deliveranceApi, "deliveranceApi");
        this.objectMapper = Objects.requireNonNull(objectMapper, "objectMapper");
        this.defaultOptions = Objects.requireNonNull(defaultOptions, "defaultOptions");
    }

    @Override
    public ChatResponse call(Prompt prompt) {
        CreateChatCompletionResponse response = deliveranceApi.createChatCompletion(toRequest(prompt, false));
        String content = response == null || response.getChoices() == null || response.getChoices().isEmpty()
                ? "" : response.getChoices().get(0).getMessage().getContent();
        return new ChatResponse(List.of(new Generation(new AssistantMessage(content))));
    }

    @Override
    public ChatOptions getOptions() {
        return defaultOptions;
    }

    CreateChatCompletionRequest toRequest(Prompt prompt, boolean stream) {
        DeliveranceChatOptions options = mergeOptions(prompt.getOptions());
        CreateChatCompletionRequest request = new CreateChatCompletionRequest()
                .model(requireModel(options))
                .stream(stream);
        List<ChatCompletionRequestMessage> messages = new ArrayList<>();
        for (Message message : prompt.getInstructions()) {
            messages.add(new ChatCompletionRequestMessage().role(role(message)).content(message.getText()));
        }
        request.messages(messages);
        if (options.getTemperature() != null) {
            request.temperature(BigDecimal.valueOf(options.getTemperature()));
        }
        if (options.getTopP() != null) {
            request.topP(BigDecimal.valueOf(options.getTopP()));
        }
        if (options.getTopK() != null) {
            request.topK(BigDecimal.valueOf(options.getTopK()));
        }
        if (options.getMaxTokens() != null) {
            request.maxTokens(options.getMaxTokens());
        }
        if (options.getSeed() != null) {
            request.seed(options.getSeed());
        }
        if (options.getLogprobs() != null) {
            request.logprobs(options.getLogprobs());
        }
        if (options.getTopLogprobs() != null) {
            request.topLogprobs(options.getTopLogprobs());
        }
        if (options.getXtcThreshold() != null) {
            request.xtcThreshold(BigDecimal.valueOf(options.getXtcThreshold()));
        }
        if (options.getXtcProbability() != null) {
            request.xtcProbability(BigDecimal.valueOf(options.getXtcProbability()));
        }
        if (options.getGuidedRegex() != null) {
            request.guidedRegex(options.getGuidedRegex());
        }
        if (options.getGuidedJson() != null) {
            try {
                request.guidedJson(objectMapper.convertValue(objectMapper.readTree(options.getGuidedJson()), Map.class));
            } catch (Exception e) {
                throw new IllegalArgumentException("guidedJson must be valid JSON schema", e);
            }
        }
        return request;
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

    private String requireModel(DeliveranceChatOptions options) {
        if (options.getModel() == null || options.getModel().isBlank()) {
            throw new IllegalArgumentException("spring.ai.deliverance.model must be set");
        }
        return options.getModel();
    }

    private String role(Message message) {
        MessageType type = message.getMessageType();
        if (type == MessageType.SYSTEM) {
            return "system";
        }
        if (type == MessageType.ASSISTANT) {
            return "assistant";
        }
        if (type == MessageType.TOOL) {
            return "tool";
        }
        return "user";
    }
}
