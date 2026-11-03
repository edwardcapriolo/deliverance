package net.reckoning.http;

import io.teknek.reckoning.generator.GeneratorParameters;
import io.teknek.reckoning.safetensors.prompt.PromptSupport;

public record PreparedRequest(PromptSupport.Builder promptSupportBuilder, GeneratorParameters generatorParameters) {
}
