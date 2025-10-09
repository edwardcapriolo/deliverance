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
     * @param temperature the temperature [0.0, 1.0]
     * @param ntokens the number of tokens to generate
     * @param onTokenWithTimings a callback for each token generated
     * @return the response
     */
    //TODO should things like temp and ntokens be in the prompt context what ofthe other zillions of pararms
    Response generate(UUID session, PromptContext promptContext, float temperature, int ntokens, Optional<Integer> seed,
            BiConsumer<String, Float> onTokenWithTimings
    );
}