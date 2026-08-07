package io.teknek.deliverance.springai;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import io.micrometer.observation.Observation;
import io.micrometer.observation.ObservationRegistry;
import io.micrometer.observation.contextpropagation.ObservationThreadLocalAccessor;
import io.teknek.deliverance.client.spring.model.ChatCompletionMessageToolCall;
import io.teknek.deliverance.client.spring.model.ChatCompletionMessageToolCallFunction;
import io.teknek.deliverance.client.spring.model.ChatCompletionRequestMessage;
import io.teknek.deliverance.client.spring.model.ChatCompletionResponseMessage;
import io.teknek.deliverance.client.spring.model.ChatCompletionTool;
import io.teknek.deliverance.client.spring.model.CreateChatCompletionRequest;
import io.teknek.deliverance.client.spring.model.CreateChatCompletionResponse;
import io.teknek.deliverance.client.spring.model.CreateChatCompletionResponseChoicesInner;
import io.teknek.deliverance.client.spring.model.FunctionObject;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.MessageType;
import org.springframework.ai.chat.messages.ToolResponseMessage;
import org.springframework.ai.chat.metadata.ChatGenerationMetadata;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.model.MessageAggregator;
import org.springframework.ai.chat.observation.ChatModelObservationContext;
import org.springframework.ai.chat.observation.ChatModelObservationConvention;
import org.springframework.ai.chat.observation.ChatModelObservationDocumentation;
import org.springframework.ai.chat.observation.DefaultChatModelObservationConvention;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.model.tool.ToolCallingManager;
import org.springframework.ai.retry.RetryUtils;
import org.springframework.ai.tool.definition.ToolDefinition;
import org.springframework.core.retry.RetryTemplate;
import org.springframework.util.Assert;
import org.springframework.util.CollectionUtils;
import org.springframework.util.StringUtils;
import reactor.core.publisher.Flux;
import tools.jackson.databind.ObjectMapper;

public class DeliveranceChatModel implements ChatModel {
    private static final Log logger = LogFactory.getLog(DeliveranceChatModel.class);
    private static final ChatModelObservationConvention DEFAULT_OBSERVATION_CONVENTION = new DefaultChatModelObservationConvention();
    private static final ToolCallingManager DEFAULT_TOOL_CALLING_MANAGER = ToolCallingManager.builder().build();

    private final DeliveranceApi deliveranceApi;
    private final ObjectMapper objectMapper;
    private final DeliveranceChatOptions defaultOptions;
    private final ToolCallingManager toolCallingManager;
    private final RetryTemplate retryTemplate;
    private final ObservationRegistry observationRegistry;
    private ChatModelObservationConvention observationConvention = DEFAULT_OBSERVATION_CONVENTION;

    public DeliveranceChatModel(DeliveranceApi deliveranceApi, ObjectMapper objectMapper,
            DeliveranceChatOptions defaultOptions) {
        this(deliveranceApi, objectMapper, defaultOptions, DEFAULT_TOOL_CALLING_MANAGER);
    }

    public DeliveranceChatModel(DeliveranceApi deliveranceApi, ObjectMapper objectMapper,
            DeliveranceChatOptions defaultOptions, ToolCallingManager toolCallingManager) {
        this(deliveranceApi, objectMapper, defaultOptions, toolCallingManager, RetryUtils.DEFAULT_RETRY_TEMPLATE,
                ObservationRegistry.NOOP);
    }

    public DeliveranceChatModel(DeliveranceApi deliveranceApi, ObjectMapper objectMapper,
            DeliveranceChatOptions defaultOptions, ToolCallingManager toolCallingManager, RetryTemplate retryTemplate,
            ObservationRegistry observationRegistry) {
        this.deliveranceApi = Objects.requireNonNull(deliveranceApi, "deliveranceApi");
        this.objectMapper = Objects.requireNonNull(objectMapper, "objectMapper");
        this.defaultOptions = Objects.requireNonNull(defaultOptions, "defaultOptions");
        this.toolCallingManager = Objects.requireNonNull(toolCallingManager, "toolCallingManager");
        this.retryTemplate = Objects.requireNonNull(retryTemplate, "retryTemplate");
        this.observationRegistry = Objects.requireNonNull(observationRegistry, "observationRegistry");
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public ChatResponse call(Prompt prompt) {
        Prompt requestPrompt = buildRequestPrompt(prompt);
        CreateChatCompletionRequest request = toRequest(requestPrompt, false);
        ChatModelObservationContext observationContext = ChatModelObservationContext.builder()
                .prompt(requestPrompt)
                .provider(DeliveranceApi.PROVIDER_NAME)
                .build();

        return ChatModelObservationDocumentation.CHAT_MODEL_OPERATION
                .observation(this.observationConvention, DEFAULT_OBSERVATION_CONVENTION, () -> observationContext,
                        this.observationRegistry)
                .observe(() -> {
                    CreateChatCompletionResponse response = RetryUtils.execute(this.retryTemplate,
                            () -> this.deliveranceApi.createChatCompletion(request));
                    ChatResponse chatResponse = toChatResponse(response, requestPrompt);
                    observationContext.setResponse(chatResponse);
                    return chatResponse;
                });
    }

    @Override
    public Flux<ChatResponse> stream(Prompt prompt) {
        Prompt requestPrompt = buildRequestPrompt(prompt);
        return Flux.deferContextual(contextView -> {
            CreateChatCompletionRequest request = toRequest(requestPrompt, true);
            ChatModelObservationContext observationContext = ChatModelObservationContext.builder()
                    .prompt(requestPrompt)
                    .provider(DeliveranceApi.PROVIDER_NAME)
                    .streaming(true)
                    .build();
            Observation observation = ChatModelObservationDocumentation.CHAT_MODEL_OPERATION.observation(
                    this.observationConvention, DEFAULT_OBSERVATION_CONVENTION, () -> observationContext,
                    this.observationRegistry);
            Observation parentObservation = contextView.getOrDefault(ObservationThreadLocalAccessor.KEY, null);
            observation.parentObservation(parentObservation);
            try (Observation.Scope ignored = parentObservation != null ? parentObservation.openScope()
                    : Observation.Scope.NOOP) {
                observation.start();
            }

            Flux<ChatResponse> responseFlux = RetryUtils.execute(this.retryTemplate,
                    () -> this.deliveranceApi.streamChatCompletion(request));
            Flux<ChatResponse> observedFlux = responseFlux.doOnError(observation::error)
                    .doFinally(s -> observation.stop())
                    .contextWrite(ctx -> ctx.put(ObservationThreadLocalAccessor.KEY, observation));
            return new MessageAggregator().aggregate(observedFlux, observationContext::setResponse);
        });
    }

    @Override
    public DeliveranceChatOptions getOptions() {
        return defaultOptions;
    }

    public void setObservationConvention(ChatModelObservationConvention observationConvention) {
        Assert.notNull(observationConvention, "observationConvention cannot be null");
        this.observationConvention = observationConvention;
    }

    CreateChatCompletionRequest toRequest(Prompt prompt, boolean stream) {
        DeliveranceChatOptions options = mergeOptions(prompt.getOptions());
        CreateChatCompletionRequest request = new CreateChatCompletionRequest()
                .model(requireModel(options))
                .stream(stream);
        List<ChatCompletionRequestMessage> messages = new ArrayList<>();
        for (Message message : prompt.getInstructions()) {
            messages.addAll(toMessages(message));
        }
        request.messages(messages);
        if (options.getTemperature() != null) {
            request.temperature(BigDecimal.valueOf(options.getTemperature()));
        }
        if (options.getTopP() != null) {
            request.topP(BigDecimal.valueOf(options.getTopP()));
        }
        if (options.getUniformTopP() != null) {
            request.uniformTopP(BigDecimal.valueOf(options.getUniformTopP()));
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
        List<ToolDefinition> toolDefinitions = this.toolCallingManager.resolveToolDefinitions(options);
        if (!CollectionUtils.isEmpty(toolDefinitions)) {
            request.tools(toolDefinitions.stream().map(this::toTool).toList());
            request.parallelToolCalls(false);
        }
        return request;
    }

    private ChatCompletionTool toTool(ToolDefinition toolDefinition) {
        try {
            @SuppressWarnings("unchecked")
            Map<String, Object> parameters = this.objectMapper.readValue(toolDefinition.inputSchema(), Map.class);
            return new ChatCompletionTool().type("function")
                    .function(new FunctionObject().name(toolDefinition.name())
                            .description(toolDefinition.description())
                            .parameters(parameters));
        }
        catch (Exception ex) {
            throw new IllegalArgumentException("Tool input schema must be valid JSON", ex);
        }
    }

    private ChatResponse toChatResponse(CreateChatCompletionResponse response, Prompt prompt) {
        if (response == null) {
            throw new IllegalStateException("Deliverance chat completion returned no response");
        }
        if (response.getChoices() == null || response.getChoices().isEmpty()) {
            if (logger.isWarnEnabled()) {
                logger.warn("No choices returned for prompt: " + prompt);
            }
            return new ChatResponse(List.of(), DeliveranceApi.metadata(response));
        }
        List<Generation> generations = response.getChoices().stream().map(this::toGeneration).toList();
        return new ChatResponse(generations, DeliveranceApi.metadata(response));
    }

    private Generation toGeneration(CreateChatCompletionResponseChoicesInner choice) {
        ChatCompletionResponseMessage responseMessage = choice.getMessage();
        String content = responseMessage != null && responseMessage.getContent() != null ? responseMessage.getContent() : "";
        AssistantMessage assistantMessage = AssistantMessage.builder()
                .content(content)
                .toolCalls(responseMessage != null ? DeliveranceApi.toolCalls(responseMessage.getToolCalls()) : List.of())
                .build();
        String finishReason = choice.getFinishReason() != null ? choice.getFinishReason().getValue() : "";
        return new Generation(assistantMessage, ChatGenerationMetadata.builder().finishReason(finishReason).build());
    }

    private List<ChatCompletionRequestMessage> toMessages(Message message) {
        if (message instanceof ToolResponseMessage toolResponseMessage) {
            return toolResponseMessage.getResponses().stream()
                    .map(response -> new ChatCompletionRequestMessage().role("tool")
                            .content(response.responseData())
                            .toolCallId(response.id()))
                    .toList();
        }
        ChatCompletionRequestMessage requestMessage = new ChatCompletionRequestMessage()
                .role(role(message))
                .content(message.getText());
        if (message instanceof AssistantMessage assistantMessage && !assistantMessage.getToolCalls().isEmpty()) {
            requestMessage.toolCalls(assistantMessage.getToolCalls().stream().map(this::toToolCall).toList());
        }
        return List.of(requestMessage);
    }

    private ChatCompletionMessageToolCall toToolCall(AssistantMessage.ToolCall toolCall) {
        return new ChatCompletionMessageToolCall().id(toolCall.id())
                .type(toolCall.type())
                .function(new ChatCompletionMessageToolCallFunction().name(toolCall.name()).arguments(toolCall.arguments()));
    }

    private Prompt buildRequestPrompt(Prompt prompt) {
        if (prompt.getOptions() == null) {
            return prompt.mutate().chatOptions(this.getOptions()).build();
        }
        return prompt;
    }

    private DeliveranceChatOptions mergeOptions(ChatOptions promptOptions) {
        if (promptOptions == null) {
            return defaultOptions;
        }
        if (promptOptions instanceof DeliveranceChatOptions deliveranceOptions) {
            return DeliveranceChatOptions.builder()
                    .model(deliveranceOptions.getModel() == null ? defaultOptions.getModel() : deliveranceOptions.getModel())
                    .temperature(deliveranceOptions.getTemperature() == null ? defaultOptions.getTemperature() : deliveranceOptions.getTemperature())
                    .maxTokens(deliveranceOptions.getMaxTokens() == null ? defaultOptions.getMaxTokens() : deliveranceOptions.getMaxTokens())
                    .topP(deliveranceOptions.getTopP() == null ? defaultOptions.getTopP() : deliveranceOptions.getTopP())
                    .uniformTopP(deliveranceOptions.getUniformTopP() == null ? defaultOptions.getUniformTopP() : deliveranceOptions.getUniformTopP())
                    .topK(deliveranceOptions.getTopK() == null ? defaultOptions.getTopK() : deliveranceOptions.getTopK())
                    .stopSequences(deliveranceOptions.getStopSequences() == null ? defaultOptions.getStopSequences() : deliveranceOptions.getStopSequences())
                    .seed(deliveranceOptions.getSeed() == null ? defaultOptions.getSeed() : deliveranceOptions.getSeed())
                    .logprobs(deliveranceOptions.getLogprobs() == null ? defaultOptions.getLogprobs() : deliveranceOptions.getLogprobs())
                    .topLogprobs(deliveranceOptions.getTopLogprobs() == null ? defaultOptions.getTopLogprobs() : deliveranceOptions.getTopLogprobs())
                    .xtcThreshold(deliveranceOptions.getXtcThreshold() == null ? defaultOptions.getXtcThreshold() : deliveranceOptions.getXtcThreshold())
                    .xtcProbability(deliveranceOptions.getXtcProbability() == null ? defaultOptions.getXtcProbability() : deliveranceOptions.getXtcProbability())
                    .guidedRegex(deliveranceOptions.getGuidedRegex() == null ? defaultOptions.getGuidedRegex() : deliveranceOptions.getGuidedRegex())
                    .guidedJson(deliveranceOptions.getGuidedJson() == null ? defaultOptions.getGuidedJson() : deliveranceOptions.getGuidedJson())
                    .toolCallbacks(deliveranceOptions.getToolCallbacks() == null ? defaultOptions.getToolCallbacks() : deliveranceOptions.getToolCallbacks())
                    .toolContext(deliveranceOptions.getToolContext() == null ? defaultOptions.getToolContext() : deliveranceOptions.getToolContext())
                    .build();
        }
        return DeliveranceChatOptions.builder()
                .model(promptOptions.getModel() == null ? defaultOptions.getModel() : promptOptions.getModel())
                .temperature(promptOptions.getTemperature() == null ? defaultOptions.getTemperature() : promptOptions.getTemperature())
                .maxTokens(promptOptions.getMaxTokens() == null ? defaultOptions.getMaxTokens() : promptOptions.getMaxTokens())
                .topP(promptOptions.getTopP() == null ? defaultOptions.getTopP() : promptOptions.getTopP())
                .uniformTopP(defaultOptions.getUniformTopP())
                .topK(promptOptions.getTopK() == null ? defaultOptions.getTopK() : promptOptions.getTopK())
                .seed(defaultOptions.getSeed())
                .logprobs(defaultOptions.getLogprobs())
                .topLogprobs(defaultOptions.getTopLogprobs())
                .xtcThreshold(defaultOptions.getXtcThreshold())
                .xtcProbability(defaultOptions.getXtcProbability())
                .guidedRegex(defaultOptions.getGuidedRegex())
                .guidedJson(defaultOptions.getGuidedJson())
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

    public static final class Builder {
        private DeliveranceApi deliveranceApi;
        private ObjectMapper objectMapper = new ObjectMapper();
        private DeliveranceChatOptions options = DeliveranceChatOptions.builder().build();
        private ToolCallingManager toolCallingManager = DEFAULT_TOOL_CALLING_MANAGER;
        private RetryTemplate retryTemplate = RetryUtils.DEFAULT_RETRY_TEMPLATE;
        private ObservationRegistry observationRegistry = ObservationRegistry.NOOP;

        private Builder() {
        }

        public Builder deliveranceApi(DeliveranceApi deliveranceApi) {
            this.deliveranceApi = deliveranceApi;
            return this;
        }

        public Builder objectMapper(ObjectMapper objectMapper) {
            Assert.notNull(objectMapper, "objectMapper cannot be null");
            this.objectMapper = objectMapper;
            return this;
        }

        public Builder options(DeliveranceChatOptions options) {
            Assert.notNull(options, "options cannot be null");
            this.options = options;
            return this;
        }

        public Builder toolCallingManager(ToolCallingManager toolCallingManager) {
            Assert.notNull(toolCallingManager, "toolCallingManager cannot be null");
            this.toolCallingManager = toolCallingManager;
            return this;
        }

        public Builder retryTemplate(RetryTemplate retryTemplate) {
            Assert.notNull(retryTemplate, "retryTemplate cannot be null");
            this.retryTemplate = retryTemplate;
            return this;
        }

        public Builder observationRegistry(ObservationRegistry observationRegistry) {
            Assert.notNull(observationRegistry, "observationRegistry cannot be null");
            this.observationRegistry = observationRegistry;
            return this;
        }

        public DeliveranceChatModel build() {
            Assert.state(this.deliveranceApi != null, "DeliveranceApi must not be null");
            return new DeliveranceChatModel(this.deliveranceApi, this.objectMapper, this.options,
                    this.toolCallingManager, this.retryTemplate, this.observationRegistry);
        }
    }
}
