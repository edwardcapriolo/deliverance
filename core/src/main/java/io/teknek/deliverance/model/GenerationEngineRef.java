package io.teknek.deliverance.model;

import com.google.common.base.Preconditions;
import io.dropwizard.metrics5.Timer;
import io.teknek.deliverance.generator.FinishReason;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.guided.LogitsProcessorFactory;
import io.teknek.deliverance.guided.LogitsProcessorRef;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor2.TensorRef;
import io.teknek.sketches.SketchesSettings;

import java.util.Arrays;
import java.util.Objects;
import java.util.Optional;
import java.util.Random;
import java.util.UUID;
import java.util.concurrent.CancellationException;
import java.util.concurrent.TimeUnit;

/**
 * Sampling boundary used by a TensorRef generation loop.
 *
 * <p>Forward execution remains owned by {@link GenerationBackendRef}; this class deliberately accepts its returned
 * {@link TensorRef} directly and never creates an {@code AbstractTensor} view.</p>
 */
public final class GenerationEngineRef {
    private final SketchesSettings sketchesSettings;

    public GenerationEngineRef() {
        this(SketchesSettings.DEFAULT);
    }

    public GenerationEngineRef(SketchesSettings sketchesSettings) {
        this.sketchesSettings = Objects.requireNonNull(sketchesSettings, "sketchesSettings");
    }

    /** Samples one native forward result; a caller-owned logits and argmax scratch are reused across steps. */
    public SamplerReturn sample(AbstractModel model, GeneratorParameters parameters, TensorRef output,
            TensorRef logits, TensorRef argMaxScratch, ResponseContext responseContext, Random random) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(model.sampleOutputRef, "model TensorRef output weights");
        Optional<LogitsProcessorRef> processor = LogitsProcessorFactory.createRef(model, parameters, sketchesSettings);
        if (output.shape().first() > 1) {
            try (TensorRef lastRow = output.slice((int) output.shape().first() - 1)) {
                return model.createNextTokenRef(parameters, lastRow, logits, responseContext, random, processor,
                        argMaxScratch);
            }
        }
        return model.createNextTokenRef(parameters, output, logits, responseContext, random, processor, argMaxScratch);
    }

    /** Runs the complete autoregressive loop against a TensorRef-native execution backend. */
    public Response generate(AbstractModel model, GenerationBackendRef backend, UUID sessionId,
            PromptContext promptContext, GeneratorParameters parameters, GenerateEvent eventFired) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(backend, "backend");
        Objects.requireNonNull(sessionId, "sessionId");
        try (Timer.Context ignoredRequest = InferenceProfiler.timer(model.getMetricRegistry(), "generation.request").time();
             AbstractModel.TensorPlanTraceScope ignoredTrace = model.openTensorPlanTrace(sessionId)) {
            long generationStartNanos = System.nanoTime();
            long timeToFirstTokenNanos = 0L;
            ResponseContext responseContext = new ResponseContext(model);
            Random random = parameters.seed.map(Random::new).orElseGet(Random::new);
            Optional<LogitsProcessorRef> processor = LogitsProcessorFactory.createRef(model, parameters, sketchesSettings);
            long[] encoded = model.encodeText(promptContext.getPrompt());
            throwIfInterrupted();
            if (encoded.length > 0 && encoded[0] == model.config.bosToken) {
                encoded = Arrays.copyOfRange(encoded, 1, encoded.length);
            }
            int ntokens = parameters.ntokens.orElse(model.config.contextLength);
            Preconditions.checkArgument(encoded.length < model.config.contextLength && encoded.length < ntokens,
                    "Prompt exceeds ntokens");
            if (ntokens > model.config.contextLength) {
                throw new GenerationException(String.format("ntokens %d exceed config length %d", ntokens,
                        model.config.contextLength));
            }
            int[] promptTokens = model.constructPromptTokens(encoded);
            int promptTokenCount = promptTokens.length;
            try (TensorRef logits = allocateLogits(model); TensorRef argMaxScratch = allocateArgMaxScratch(model);
                 GenerationBackendRef.GenerationSessionRef session = backend.openRef(sessionId, promptTokens, parameters)) {
                GenerationCursor cursor = GenerationCursor.from(promptTokens, session.prefixLength());
                TensorRef prefillOutput;
                try (Timer.Context ignoredPrefill = InferenceProfiler.timer(model.getMetricRegistry(), "generation.prefill").time()) {
                    prefillOutput = session.prefill(cursor);
                }
                SamplerReturn nextSample;
                try (TensorRef output = prefillOutput;
                     Timer.Context ignoredSample = InferenceProfiler.timer(model.getMetricRegistry(), "generation.first_sample").time()) {
                    if (output.shape().first() > 1) {
                        try (TensorRef sampleOutput = output.slice((int) output.shape().first() - 1)) {
                            nextSample = sample(model, parameters, sampleOutput, logits, argMaxScratch, responseContext,
                                    random, processor);
                        }
                    } else {
                        nextSample = sample(model, parameters, output, logits, argMaxScratch, responseContext, random,
                                processor);
                    }
                }
                int next = nextSample.token;
                responseContext.add(nextSample, eventFired);
                timeToFirstTokenNanos = System.nanoTime() - generationStartNanos;
                model.getMetricRegistry().timer("generation.time_to_first_token").update(timeToFirstTokenNanos,
                        TimeUnit.NANOSECONDS);
                Optional<Response> firstStop = maybeStopAfterToken(model, parameters, responseContext, promptTokenCount,
                        next, generationStartNanos, timeToFirstTokenNanos);
                if (firstStop.isPresent()) {
                    return withGenerationTiming(model, firstStop.get(), generationStartNanos, timeToFirstTokenNanos);
                }
                for (int i = cursor.decodeStartPosition(); i < ntokens; i++) {
                    throwIfInterrupted();
                    SamplerReturn current;
                    try (Timer.Context ignoredDecode = InferenceProfiler.timer(model.getMetricRegistry(), "generation.decode").time();
                         TensorRef output = session.decode(next, i)) {
                        try (Timer.Context ignoredDecodeSample = InferenceProfiler.timer(model.getMetricRegistry(), "generation.decode_sample").time()) {
                            current = sample(model, parameters, output, logits, argMaxScratch, responseContext, random, processor);
                        }
                    }
                    next = current.token;
                    session.afterDecode();
                    responseContext.add(current, eventFired);
                    Optional<Response> stop = maybeStopAfterToken(model, parameters, responseContext, promptTokenCount,
                            next, generationStartNanos, timeToFirstTokenNanos);
                    if (stop.isPresent()) {
                        return withGenerationTiming(model, stop.get(), generationStartNanos, timeToFirstTokenNanos);
                    }
                }
            }
            return withGenerationTiming(model, model.postProcessResponse(new Response(
                    responseContext.responseText.toString(), responseContext.responseTextWithSpecialTokens.toString(),
                    FinishReason.MAX_TOKENS, promptTokenCount, responseContext.generatedTokens, 0, 0,
                    responseContext.samplerReturnList)), generationStartNanos, timeToFirstTokenNanos);
        }
    }

    private SamplerReturn sample(AbstractModel model, GeneratorParameters parameters, TensorRef output,
            TensorRef logits, TensorRef argMaxScratch, ResponseContext responseContext, Random random,
            Optional<LogitsProcessorRef> processor) {
        Objects.requireNonNull(model.sampleOutputRef, "model TensorRef output weights");
        return model.createNextTokenRef(parameters, output, logits, responseContext, random, processor, argMaxScratch);
    }

    private Optional<Response> maybeStopAfterToken(AbstractModel model, GeneratorParameters parameters,
            ResponseContext context, int promptLength, int next, long start, long ttft) {
        if (parameters.maxTokens.isPresent() && context.generatedTokens.size() >= parameters.maxTokens.get()) {
            return Optional.of(buildTimedResponse(model, FinishReason.MAX_TOKENS, promptLength, context, start, ttft));
        }
        if (parameters.guidedChoice.isPresent() && parameters.guidedChoice.get().contains(context.responseText.toString())) {
            return Optional.of(buildTimedResponse(model, FinishReason.STOP_TOKEN, promptLength, context, start, ttft));
        }
        Optional<Response> stop = model.stopWords(parameters, context, promptLength);
        if (stop.isPresent()) return Optional.of(model.postProcessResponse(withGenerationTiming(model, stop.get(), start, ttft)));
        Optional<Response> tools = model.getToolCallParser().shouldEndTurn(context, promptLength);
        if (tools.isPresent()) return Optional.of(model.postProcessResponse(withGenerationTiming(model, tools.get(), start, ttft)));
        return model.config.eosTokens.contains(next)
                ? Optional.of(buildTimedResponse(model, FinishReason.STOP_TOKEN, promptLength, context, start, ttft))
                : Optional.empty();
    }

    private Response buildTimedResponse(AbstractModel model, FinishReason reason, int promptLength,
            ResponseContext context, long start, long ttft) {
        return model.postProcessResponse(withGenerationTiming(model, new Response(context.responseText.toString(),
                context.responseTextWithSpecialTokens.toString(), reason, promptLength, context.generatedTokens, 0, 0,
                context.samplerReturnList), start, ttft));
    }

    private Response withGenerationTiming(AbstractModel model, Response response, long start, long ttft) {
        double totalMs = (System.nanoTime() - start) / 1_000_000.0;
        double ttftMs = ttft / 1_000_000.0;
        int count = response.generatedTokens == null ? 0 : response.generatedTokens.size();
        double avgMs = count == 0 ? 0.0 : totalMs / count;
        return response.copyWithTiming(ttftMs, avgMs, totalMs);
    }

    private static void throwIfInterrupted() {
        if (Thread.interrupted()) throw new CancellationException("generation interrupted");
    }

    /** Allocates the reusable native sampler buffers through the model's Lighter allocator. */
    public TensorRef allocateLogits(AbstractModel model) {
        return model.getLighter().allocate(io.teknek.deliverance.DType.F32,
                TensorShape.of(1, model.getConfig().vocabularySize));
    }

    public TensorRef allocateArgMaxScratch(AbstractModel model) {
        return model.getLighter().allocate(io.teknek.deliverance.DType.F32, TensorShape.of(1, 2));
    }
}
