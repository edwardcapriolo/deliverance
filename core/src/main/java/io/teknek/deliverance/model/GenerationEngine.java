package io.teknek.deliverance.model;

import com.google.common.base.Preconditions;
import io.teknek.deliverance.generator.FinishReason;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.tensor.AbstractTensor;

import java.util.Arrays;
import java.util.Optional;
import java.util.Random;
import java.util.UUID;
import java.util.concurrent.TimeUnit;

/**
 * Runs the shared autoregressive generation loop for causal language models.
 *
 * <p>This class owns request-level generation behavior: prompt encoding and validation, prompt cursor setup, token
 * sampling, stop-condition handling, event emission, response timing, and model-specific response post-processing. It
 * deliberately does not own transformer execution or KV/session storage. Those mechanics are supplied by a
 * {@link GenerationBackend} so the same loop can run against a local model, tensor-parallel ranks, or another backend.</p>
 */
final class GenerationEngine {
    /**
     * Generates a response for one rendered prompt using the supplied execution backend.
     *
     * <p>Preconditions: {@code promptContext} must contain the final text to feed the model, not structured chat
     * messages; {@code backend.open(...)} must return a session whose prefix length is valid for the constructed prompt
     * tokens; and backend tensors returned from prefill/decode must be compatible with the coordinator model's output
     * projection. The method closes backend sessions and temporary tensors it owns.</p>
     */
    Response generate(AbstractModel model, GenerationBackend backend, UUID sessionId, PromptContext promptContext,
            GeneratorParameters generatorParameters, GenerateEvent eventFired) {
        long generationStartNanos = System.nanoTime();
        long timeToFirstTokenNanos = 0L;

        ResponseContext responseContext = new ResponseContext(model);
        Random random = generatorParameters.seed.map(Random::new).orElseGet(Random::new);
        long[] encoded = model.encodeText(promptContext.getPrompt());
        if (encoded.length > 0 && encoded[0] == model.config.bosToken) {
            AbstractModel.logger.warn("encoded [] started with BOS token removing it");
            encoded = Arrays.copyOfRange(encoded, 1, encoded.length);
        }
        int ntokens = generatorParameters.ntokens.orElse(model.config.contextLength);
        Preconditions.checkArgument(encoded.length < model.config.contextLength
                && encoded.length < ntokens, "Prompt exceeds ntokens");
        if (ntokens > model.config.contextLength) {
            throw new GenerationException(String.format("ntokens %d exceed config length %d", ntokens,
                    model.config.contextLength));
        }
        float temperature = generatorParameters.temperature.orElse(0.0f);
        int[] promptTokens = model.constructPromptTokens(encoded);
        int promptTokenCount = promptTokens.length;
        try (AbstractTensor logits = model.makeDenseTensor(model.config.vocabularySize)) {
            try (GenerationBackend.GenerationSession session = backend.open(sessionId, promptTokens, generatorParameters)) {
                GenerationCursor cursor = GenerationCursor.from(promptTokens, session.prefixLength());
                AbstractTensor last = session.prefill(cursor);
                SamplerReturn nextSamplerRet = model.createNextToken(generatorParameters, logits, last, responseContext,
                        random, temperature);
                int next = nextSamplerRet.token;
                last.close();
                responseContext.add(nextSamplerRet, eventFired);
                timeToFirstTokenNanos = System.nanoTime() - generationStartNanos;
                model.metricRegistry.timer("generation.time_to_first_token").update(timeToFirstTokenNanos,
                        TimeUnit.NANOSECONDS);
                AbstractModel.logger.info("time_to_first_token={} prefix_length={}",
                        timeToFirstTokenNanos / 1_000_000.0, cursor.prefixLength());
                Optional<Response> firstStop = maybeStopAfterToken(model, generatorParameters, responseContext,
                        promptTokenCount, next, generationStartNanos, timeToFirstTokenNanos);
                if (firstStop.isPresent()) {
                    return withGenerationTiming(model, firstStop.get(), generationStartNanos, timeToFirstTokenNanos);
                }
                for (int i = cursor.decodeStartPosition(); i < ntokens; i++) {
                    AbstractTensor output = session.decode(next, i);
                    SamplerReturn nextSample = model.createNextTokenLoop(generatorParameters, output, logits,
                            responseContext, random, temperature);
                    next = nextSample.token;
                    output.close();
                    session.afterDecode();
                    responseContext.add(nextSample, eventFired);

                    Optional<Response> stop = maybeStopAfterToken(model, generatorParameters, responseContext,
                            promptTokenCount, next, generationStartNanos, timeToFirstTokenNanos);
                    if (stop.isPresent()) {
                        return withGenerationTiming(model, stop.get(), generationStartNanos, timeToFirstTokenNanos);
                    }
                }
            }
        }

        return withGenerationTiming(model, model.postProcessResponse(new Response(
                        responseContext.responseText.toString(),
                        responseContext.responseTextWithSpecialTokens.toString(),
                        FinishReason.MAX_TOKENS,
                        promptTokenCount,
                        responseContext.generatedTokens,
                        0,
                        0,
                        responseContext.samplerReturnList)),
                generationStartNanos,
                timeToFirstTokenNanos);
    }

    /**
     * Evaluates all stop conditions after a token has been appended to {@code responseContext}.
     *
     * <p>The checks are ordered by explicit request limits first, then guided-choice completion, stop strings, tool-call
     * parser termination, and finally model EOS tokens. When a stop condition fires, the returned response already has
     * response text, finish reason, generated tokens, sampler returns, and timing populated. If no stop condition applies,
     * returns {@link Optional#empty()} and generation should continue.</p>
     *
     * <p>Precondition: {@code responseContext} must already include {@code next}; this method inspects accumulated text
     * and generated token count rather than predicting the effect of a future token.</p>
     */
    private Optional<Response> maybeStopAfterToken(AbstractModel model, GeneratorParameters generatorParameters,
            ResponseContext responseContext, int promptLength, int next, long generationStartNanos,
            long timeToFirstTokenNanos) {
        if (generatorParameters.maxTokens.isPresent()) {
            if (responseContext.generatedTokens.size() >= generatorParameters.maxTokens.get()) {
                return Optional.of(buildTimedResponse(model, FinishReason.MAX_TOKENS, promptLength, responseContext,
                        generationStartNanos, timeToFirstTokenNanos));
            }
        }
        if (generatorParameters.guidedChoice.isPresent()) {
            if (generatorParameters.guidedChoice.get().contains(responseContext.responseText.toString())) {
                return Optional.of(buildTimedResponse(model, FinishReason.STOP_TOKEN, promptLength, responseContext,
                        generationStartNanos, timeToFirstTokenNanos));
            }
        }
        Optional<Response> shouldEnd = model.stopWords(generatorParameters, responseContext, promptLength);
        if (shouldEnd.isPresent()) {
            return Optional.of(model.postProcessResponse(withGenerationTiming(model, shouldEnd.get(), generationStartNanos,
                    timeToFirstTokenNanos)));
        }
        Optional<Response> shouldEndTools = model.getToolCallParser().shouldEndTurn(responseContext, promptLength);
        if (shouldEndTools.isPresent()) {
            return Optional.of(model.postProcessResponse(withGenerationTiming(model, shouldEndTools.get(),
                    generationStartNanos, timeToFirstTokenNanos)));
        }
        if (model.config.eosTokens.contains(next)) {
            return Optional.of(buildTimedResponse(model, FinishReason.STOP_TOKEN, promptLength, responseContext,
                    generationStartNanos, timeToFirstTokenNanos));
        }
        return Optional.empty();
    }

    /** Builds a final response for a stop condition and applies model-specific post-processing. */
    private Response buildTimedResponse(AbstractModel model, FinishReason reason, int promptLength,
            ResponseContext responseContext, long generationStartNanos, long timeToFirstTokenNanos) {
        return model.postProcessResponse(withGenerationTiming(model, new Response(
                        responseContext.responseText.toString(),
                        responseContext.responseTextWithSpecialTokens.toString(),
                        reason,
                        promptLength,
                        responseContext.generatedTokens,
                        0,
                        0,
                        responseContext.samplerReturnList),
                generationStartNanos,
                timeToFirstTokenNanos));
    }

    /**
     * Copies generation timing into a response.
     *
     * <p>{@code generationStartNanos} is the start of the whole request. {@code timeToFirstTokenNanos} is a duration, not
     * an absolute timestamp. Average token time is computed over generated token count and is zero for empty outputs.</p>
     */
    private Response withGenerationTiming(AbstractModel model, Response response, long generationStartNanos,
            long timeToFirstTokenNanos) {
        double totalTimeMs = (System.nanoTime() - generationStartNanos) / 1_000_000.0;
        double timeToFirstTokenMs = timeToFirstTokenNanos / 1_000_000.0;
        int generatedTokenCount = response.generatedTokens == null ? 0 : response.generatedTokens.size();
        double avgTimePerTokenMs = generatedTokenCount == 0 ? 0.0 : totalTimeMs / generatedTokenCount;
        return response.copyWithTiming(timeToFirstTokenMs, avgTimePerTokenMs, totalTimeMs);
    }
}
