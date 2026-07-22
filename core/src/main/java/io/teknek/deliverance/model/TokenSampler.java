package io.teknek.deliverance.model;

import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.guided.LogitsProcessor;
import io.teknek.deliverance.tensor.AbstractTensor;

import java.util.Optional;
import java.util.Random;

interface TokenSampler {
    SamplerReturn firstToken(GeneratorParameters parameters, GenerationEngine.Logits logits,
            GenerationEngine.PrefillOutput prefill, ResponseContext responseContext, Random random, float temperature,
            Optional<LogitsProcessor> logitsProcessor);

    SamplerReturn nextToken(GeneratorParameters parameters, AbstractTensor output, AbstractTensor logits,
            ResponseContext responseContext, Random random, float temperature,
            Optional<LogitsProcessor> logitsProcessor);
}
