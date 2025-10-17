package io.teknek.deliverance.generator;

import io.teknek.deliverance.safetensors.prompt.PromptContext;

import java.io.Closeable;
import java.util.Optional;
import java.util.UUID;
import java.util.function.BiConsumer;

public interface Generator extends Closeable {

    /**
     * Generate tokens from a prompt
     *
     * @param session the session id
     * @param promptContext the prompt context
     * @param onTokenWithTimings a callback for each token generated
     * @return the response
     */
    Response generate(UUID session, PromptContext promptContext, GeneratorParameters generatorParameters,
            BiConsumer<String, Float> onTokenWithTimings
    );
}