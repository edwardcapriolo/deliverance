package io.teknek.deliverance.model;

import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.guided.LogitsProcessor;
import io.teknek.deliverance.tensor.AbstractTensor;

import java.util.Objects;
import java.util.Optional;
import java.util.Random;

final class ModelTokenSampler implements TokenSampler {
    private final AbstractModel model;

    ModelTokenSampler(AbstractModel model) {
        this.model = Objects.requireNonNull(model, "model");
    }

    @Override
    public SamplerReturn firstToken(GeneratorParameters parameters, GenerationEngine.Logits logits,
            GenerationEngine.PrefillOutput prefill, ResponseContext responseContext, Random random, float temperature,
            Optional<LogitsProcessor> logitsProcessor) {
        return model.createNextToken(parameters, logits, prefill, responseContext, random, temperature, logitsProcessor);
    }

    @Override
    public SamplerReturn nextToken(GeneratorParameters parameters, AbstractTensor output, AbstractTensor logits,
            ResponseContext responseContext, Random random, float temperature,
            Optional<LogitsProcessor> logitsProcessor) {
        return model.createNextTokenLoop(parameters, output, logits, responseContext, random, temperature, logitsProcessor);
    }
}
