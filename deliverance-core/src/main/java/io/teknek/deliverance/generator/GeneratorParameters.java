package io.teknek.deliverance.generator;

import java.util.Optional;

/**
 * This class hopefully wont stay long, the prompt context doesnt hold all the possible request parameters at the moment
 * when we get closer to a full ChatCompletionRequest we can look at this
 */
public class GeneratorParameters {
    public Optional<Float> temperature = Optional.of(0.0f);
    public Optional<Integer> ntokens = Optional.of(256);
    public Optional<Integer> seed = Optional.of(42);
    public GeneratorParameters withSeed(int seed){
        this.seed = Optional.of(seed);
        return this;
    }
}
