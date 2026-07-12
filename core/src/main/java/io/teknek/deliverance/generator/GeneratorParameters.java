package io.teknek.deliverance.generator;

import java.util.List;
import java.util.Optional;

/**
 * The parameters of the request that are not part of the PromptSupport. For example a field like stopWords isnt
 * something the PrompyTemplate and jinja support, this is something the code of the inference engine supports. Thus
 * it belongs here
 */
public class GeneratorParameters {
    public Optional<Float> temperature = Optional.of(0.0f);
    public Optional<Integer> ntokens = Optional.empty();
    public Optional<Integer> seed = Optional.of(42);
    public Optional<String> cacheSalt = Optional.of("");
    public Optional<List<String>> stopWords = Optional.empty();
    public Optional<Boolean> includeStopStrInOutput = Optional.empty();
    public Optional<List<String>> guidedChoice = Optional.empty();
    public Optional<String> guidedRegex = Optional.empty();
    public Optional<Integer> maxTokens = Optional.empty();
    public Optional<Boolean> logProbs = Optional.empty();
    public Optional<Integer> topLogProbs = Optional.empty();
    public Optional<Float> xtcThreshold = Optional.empty();
    public Optional<Float> xtcProbability = Optional.empty();
    public Optional<Float> topK = Optional.empty();
    public Optional<Float> topP = Optional.empty();

    public GeneratorParameters withSeed(int seed){
        this.seed = Optional.of(seed);
        return this;
    }
    public GeneratorParameters withNtokens(int tokens){
        ntokens = Optional.of(tokens);
        return this;
    }
    public GeneratorParameters withGuidedChoice(List<String> choices){
        guidedChoice = Optional.of(choices);
        return this;
    }

    public GeneratorParameters withGuidedRegex(String regex){
        guidedRegex = Optional.of(regex);
        return this;
    }

    public GeneratorParameters withCacheSalt(String salt){
        cacheSalt = Optional.of(salt);
        return this;
    }
    public GeneratorParameters withTemperature(float tmp){
        this.temperature = Optional.of(tmp);
        return this;
    }

    public GeneratorParameters withStopWords(List<String> stopWords){
        this.stopWords = Optional.of(stopWords);
        return this;
    }

    public GeneratorParameters withIncludeStopStrInOutput(boolean include){
        this.includeStopStrInOutput= Optional.of(include);
        return this;
    }

    public GeneratorParameters withMaxTokens(int maxTokens){
        this.maxTokens = Optional.of(maxTokens);
        return this;
    }

    public GeneratorParameters withLogProbs(boolean logProbs){
        this.logProbs = Optional.of(logProbs);
        return this;
    }

    public GeneratorParameters withTopLogProbs(int topLogProbs){
        this.topLogProbs = Optional.of(topLogProbs);
        return this;
    }

    public GeneratorParameters withXtcThreshold(float threshold){
        this.xtcThreshold =  Optional.of(threshold);
        return this;
    }

    public GeneratorParameters withXtcProbability(float prob){
        this.xtcProbability = Optional.of(prob);
        return this;
    }
    /**
     * Sets top-k sampling. Values below 1.0 use the legacy fractional cutoff mode; values greater than or equal to
     * 1.0 are interpreted as an absolute candidate count, e.g. 64 keeps the top 64 tokens.
     */
    public GeneratorParameters withTopK(float topK){
        this.topK = Optional.of(topK);
        return this;
    }
    public GeneratorParameters withTopP(float topP){
        this.topP = Optional.of(topP);
        return this;
    }
}
