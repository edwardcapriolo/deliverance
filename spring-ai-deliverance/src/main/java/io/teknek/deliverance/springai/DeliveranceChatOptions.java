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
    private Double xtcThreshold;
    private Double xtcProbability;
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
    public Builder mutate() {
        return builder()
                .model(model)
                .temperature(temperature)
                .maxTokens(maxTokens)
                .topP(topP)
                .topK(topK)
                .stopSequences(stopSequences)
                .seed(seed)
                .logprobs(logprobs)
                .topLogprobs(topLogprobs)
                .xtcThreshold(xtcThreshold)
                .xtcProbability(xtcProbability)
                .guidedChoice(guidedChoice)
                .guidedRegex(guidedRegex)
                .guidedJson(guidedJson);
    }

    public static final class Builder implements ChatOptions.Builder<Builder> {
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
            if (otherOptions.getTopK() != null) {
                topK(otherOptions.getTopK());
            }
            if (otherOptions.getStopSequences() != null) {
                stopSequences(otherOptions.getStopSequences());
            }
            return this;
        }
    }
}
