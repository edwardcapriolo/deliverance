package io.teknek.deliverance.springai;

import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.model.tool.ToolCallingChatOptions;
import org.springframework.ai.tool.ToolCallback;

import java.util.List;
import java.util.Map;

public class DeliveranceChatOptions implements ToolCallingChatOptions {
    private String model;
    private Double temperature;
    private Integer maxTokens;
    private Double topP;
    private Double uniformTopP;
    private Integer topK;
    private List<String> stopSequences;
    private Integer seed;
    private Boolean logprobs;
    private Integer topLogprobs;
    private Double xtcThreshold;
    private Double xtcProbability;
    private List<String> guidedChoice;
    private String guidedRegex;
    private String guidedJson;
    private List<ToolCallback> toolCallbacks;
    private Map<String, Object> toolContext;

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public String getModel() {
        return model;
    }

    @Override
    public Double getFrequencyPenalty() {
        return null;
    }

    @Override
    public Integer getMaxTokens() {
        return maxTokens;
    }

    @Override
    public Double getPresencePenalty() {
        return null;
    }

    @Override
    public List<String> getStopSequences() {
        return stopSequences;
    }

    @Override
    public Double getTemperature() {
        return temperature;
    }

    @Override
    public Integer getTopK() {
        return topK;
    }

    @Override
    public Double getTopP() {
        return topP;
    }

    public Double getUniformTopP() {
        return uniformTopP;
    }

    public Integer getSeed() {
        return seed;
    }

    public Boolean getLogprobs() {
        return logprobs;
    }

    public Integer getTopLogprobs() {
        return topLogprobs;
    }

    public Double getXtcThreshold() {
        return xtcThreshold;
    }

    public Double getXtcProbability() {
        return xtcProbability;
    }

    public List<String> getGuidedChoice() {
        return guidedChoice;
    }

    public String getGuidedRegex() {
        return guidedRegex;
    }

    public String getGuidedJson() {
        return guidedJson;
    }

    @Override
    public List<ToolCallback> getToolCallbacks() {
        return toolCallbacks;
    }

    @Override
    public Map<String, Object> getToolContext() {
        return toolContext;
    }

    @Override
    public Builder mutate() {
        return builder()
                .model(model)
                .temperature(temperature)
                .maxTokens(maxTokens)
                .topP(topP)
                .uniformTopP(uniformTopP)
                .topK(topK)
                .stopSequences(stopSequences)
                .seed(seed)
                .logprobs(logprobs)
                .topLogprobs(topLogprobs)
                .xtcThreshold(xtcThreshold)
                .xtcProbability(xtcProbability)
                .guidedChoice(guidedChoice)
                .guidedRegex(guidedRegex)
                .guidedJson(guidedJson)
                .toolCallbacks(toolCallbacks)
                .toolContext(toolContext);
    }

    public static final class Builder implements ToolCallingChatOptions.Builder<Builder> {
        private final DeliveranceChatOptions options = new DeliveranceChatOptions();

        @Override
        public Builder clone() {
            return options.mutate();
        }

        @Override
        public Builder model(String model) {
            options.model = model;
            return this;
        }

        @Override
        public Builder frequencyPenalty(Double frequencyPenalty) {
            return this;
        }

        @Override
        public Builder temperature(Double temperature) {
            options.temperature = temperature;
            return this;
        }

        @Override
        public Builder maxTokens(Integer maxTokens) {
            options.maxTokens = maxTokens;
            return this;
        }

        @Override
        public Builder presencePenalty(Double presencePenalty) {
            return this;
        }

        @Override
        public Builder topP(Double topP) {
            options.topP = topP;
            return this;
        }

        public Builder uniformTopP(Double uniformTopP) {
            options.uniformTopP = uniformTopP;
            return this;
        }

        @Override
        public Builder topK(Integer topK) {
            options.topK = topK;
            return this;
        }

        @Override
        public Builder stopSequences(List<String> stopSequences) {
            options.stopSequences = stopSequences == null ? null : List.copyOf(stopSequences);
            return this;
        }

        public Builder seed(Integer seed) {
            options.seed = seed;
            return this;
        }

        public Builder logprobs(Boolean logprobs) {
            options.logprobs = logprobs;
            return this;
        }

        public Builder topLogprobs(Integer topLogprobs) {
            options.topLogprobs = topLogprobs;
            return this;
        }

        public Builder xtcThreshold(Double xtcThreshold) {
            options.xtcThreshold = xtcThreshold;
            return this;
        }

        public Builder xtcProbability(Double xtcProbability) {
            options.xtcProbability = xtcProbability;
            return this;
        }

        public Builder guidedChoice(List<String> guidedChoice) {
            options.guidedChoice = guidedChoice == null ? null : List.copyOf(guidedChoice);
            return this;
        }

        public Builder guidedRegex(String guidedRegex) {
            options.guidedRegex = guidedRegex;
            return this;
        }

        public Builder guidedJson(String guidedJson) {
            options.guidedJson = guidedJson;
            return this;
        }

        @Override
        public Builder toolCallbacks(List<ToolCallback> toolCallbacks) {
            options.toolCallbacks = toolCallbacks == null ? null : List.copyOf(toolCallbacks);
            return this;
        }

        @Override
        public Builder toolCallbacks(ToolCallback... toolCallbacks) {
            options.toolCallbacks = toolCallbacks == null ? null : List.of(toolCallbacks);
            return this;
        }

        @Override
        public Builder toolContext(Map<String, Object> context) {
            options.toolContext = context == null ? null : Map.copyOf(context);
            return this;
        }

        @Override
        public Builder toolContext(String key, Object value) {
            options.toolContext = options.toolContext == null ? new java.util.HashMap<>()
                    : new java.util.HashMap<>(options.toolContext);
            options.toolContext.put(key, value);
            return this;
        }

        @Override
        public DeliveranceChatOptions build() {
            return options;
        }

        @Override
        public Builder combineWith(ChatOptions.Builder<?> other) {
            ChatOptions otherOptions = other.build();
            if (otherOptions.getModel() != null) {
                model(otherOptions.getModel());
            }
            if (otherOptions.getTemperature() != null) {
                temperature(otherOptions.getTemperature());
            }
            if (otherOptions.getMaxTokens() != null) {
                maxTokens(otherOptions.getMaxTokens());
            }
            if (otherOptions.getTopP() != null) {
                topP(otherOptions.getTopP());
            }
            if (otherOptions instanceof DeliveranceChatOptions deliveranceOptions
                    && deliveranceOptions.getUniformTopP() != null) {
                uniformTopP(deliveranceOptions.getUniformTopP());
            }
            if (otherOptions.getTopK() != null) {
                topK(otherOptions.getTopK());
            }
            if (otherOptions.getStopSequences() != null) {
                stopSequences(otherOptions.getStopSequences());
            }
            if (otherOptions instanceof ToolCallingChatOptions toolCallingChatOptions) {
                if (toolCallingChatOptions.getToolCallbacks() != null) {
                    toolCallbacks(toolCallingChatOptions.getToolCallbacks());
                }
                if (toolCallingChatOptions.getToolContext() != null) {
                    toolContext(toolCallingChatOptions.getToolContext());
                }
            }
            return this;
        }
    }
}
