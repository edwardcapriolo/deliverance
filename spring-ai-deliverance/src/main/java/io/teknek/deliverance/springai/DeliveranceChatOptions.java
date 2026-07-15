package io.teknek.deliverance.springai;

import org.springframework.ai.chat.prompt.ChatOptions;

import java.util.List;

public class DeliveranceChatOptions implements ChatOptions {
    private String model;
    private Double temperature;
    private Integer maxTokens;
    private Double topP;
    private Integer topK;
    private List<String> stopSequences;
    private Integer seed;
    private Boolean logprobs;
    private Integer topLogprobs;
    private List<String> guidedChoice;
    private String guidedRegex;
    private String guidedJson;

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

    public Integer getSeed() {
        return seed;
    }

    public Boolean getLogprobs() {
        return logprobs;
    }

    public Integer getTopLogprobs() {
        return topLogprobs;
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
    @SuppressWarnings("unchecked")
    public <T extends ChatOptions> T copy() {
        DeliveranceChatOptions copy = new DeliveranceChatOptions();
        copy.model = model;
        copy.temperature = temperature;
        copy.maxTokens = maxTokens;
        copy.topP = topP;
        copy.topK = topK;
        copy.stopSequences = stopSequences == null ? null : List.copyOf(stopSequences);
        copy.seed = seed;
        copy.logprobs = logprobs;
        copy.topLogprobs = topLogprobs;
        copy.guidedChoice = guidedChoice == null ? null : List.copyOf(guidedChoice);
        copy.guidedRegex = guidedRegex;
        copy.guidedJson = guidedJson;
        return (T) copy;
    }

    public static final class Builder {
        private final DeliveranceChatOptions options = new DeliveranceChatOptions();

        public Builder model(String model) {
            options.model = model;
            return this;
        }

        public Builder temperature(Double temperature) {
            options.temperature = temperature;
            return this;
        }

        public Builder maxTokens(Integer maxTokens) {
            options.maxTokens = maxTokens;
            return this;
        }

        public Builder topP(Double topP) {
            options.topP = topP;
            return this;
        }

        public Builder topK(Integer topK) {
            options.topK = topK;
            return this;
        }

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

        public DeliveranceChatOptions build() {
            return options;
        }
    }
}
