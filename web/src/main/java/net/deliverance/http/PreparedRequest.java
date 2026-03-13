package net.deliverance.http;

import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;

public record PreparedRequest(PromptSupport.Builder promptSupportBuilder, GeneratorParameters generatorParameters) {
}
