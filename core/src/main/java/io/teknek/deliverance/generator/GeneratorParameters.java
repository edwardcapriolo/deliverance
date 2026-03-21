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
    public Optional<Integer> ntokens = Optional.of(256);
    public Optional<Integer> seed = Optional.of(42);
    //public Optional<String> cacheSalt = Optional.of("sha1obetter");
    public Optional<List<String>> stopWords = Optional.empty();
    public Optional<Boolean> includeStopStrInOutput = Optional.empty();
    public Optional<List<String>> guidedChoice = Optional.empty();
    public Optional<Integer> maxTokens = Optional.empty();

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

    //public GeneratorParameters withSalt(String salt){
    //    cacheSalt = Optional.of(salt);
    //    return this;
    //}
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
}
