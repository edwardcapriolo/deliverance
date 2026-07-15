package io.teknek.deliverance.springai;

import io.teknek.deliverance.generator.GeneratorParameters;
import org.springframework.ai.chat.prompt.ChatOptions;

final class DeliveranceOptionsMapper {
    private DeliveranceOptionsMapper() {
    }

    static GeneratorParameters toGeneratorParameters(ChatOptions options) {
        GeneratorParameters parameters = new GeneratorParameters();
        if (options == null) {
            return parameters;
        }
        if (options.getTemperature() != null) {
            parameters.withTemperature(options.getTemperature().floatValue());
        }
        if (options.getMaxTokens() != null) {
            parameters.withMaxTokens(options.getMaxTokens());
        }
        if (options.getTopP() != null) {
            parameters.withTopP(options.getTopP().floatValue());
        }
        if (options.getTopK() != null) {
            parameters.withTopK(options.getTopK());
        }
        if (options.getStopSequences() != null) {
            parameters.withStopWords(options.getStopSequences());
        }
        if (options instanceof DeliveranceChatOptions deliveranceOptions) {
            if (deliveranceOptions.getSeed() != null) {
                parameters.withSeed(deliveranceOptions.getSeed());
            }
            if (deliveranceOptions.getLogprobs() != null) {
                parameters.withLogProbs(deliveranceOptions.getLogprobs());
            }
            if (deliveranceOptions.getTopLogprobs() != null) {
                parameters.withTopLogProbs(deliveranceOptions.getTopLogprobs());
            }
            if (deliveranceOptions.getGuidedChoice() != null) {
                parameters.withGuidedChoice(deliveranceOptions.getGuidedChoice());
            }
            if (deliveranceOptions.getGuidedRegex() != null) {
                parameters.withGuidedRegex(deliveranceOptions.getGuidedRegex());
            }
            if (deliveranceOptions.getGuidedJson() != null) {
                parameters.withGuidedJson(deliveranceOptions.getGuidedJson());
            }
        }
        return parameters;
    }
}
