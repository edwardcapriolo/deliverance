package io.teknek.reckoning.model;

import io.teknek.reckoning.generator.GeneratorParameters;
import io.teknek.reckoning.generator.Response;
import io.teknek.reckoning.grace.PreTrainedTokenizer;
import io.teknek.reckoning.safetensors.Config;
import io.teknek.reckoning.safetensors.LoraAdapter;
import io.teknek.reckoning.safetensors.fetch.LoraAdapterModelFetcher;
import io.teknek.reckoning.safetensors.prompt.PromptContext;
import io.teknek.reckoning.safetensors.prompt.PromptSupport;
import io.teknek.reckoning.toolcallparser.ToolCallParser;
import io.teknek.sketches.SketchesSettings;
import io.github.resilience4j.timelimiter.TimeLimiter;

import java.io.IOException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeoutException;
import java.util.Objects;
import java.util.Optional;
import java.util.UUID;

/**
 * Standard {@link CausalLanguageModel} returned by the causal-LM loader.
 *
 * <p>The facade exposes the task-level API while delegating transformer execution to a {@link GenerationBackendRef}. The
 * same facade can therefore serve a normal local model or a distributed/tensor-parallel runtime without changing caller
 * code. The coordinator model supplies tokenizer, prompt support, output projection, sampling helpers, and
 * model-specific response post-processing.</p>
 */
public final class DefaultCausalLanguageModel implements CausalLanguageModel {
    private final AbstractModel coordinatorModel;
    private final GenerationBackendRef backend;

    public DefaultCausalLanguageModel(AbstractModel coordinatorModel, GenerationBackendRef backend) {
        this(coordinatorModel, backend, SketchesSettings.DEFAULT);
    }

    public DefaultCausalLanguageModel(AbstractModel coordinatorModel, GenerationBackendRef backend,
            SketchesSettings sketchesSettings) {
        this.coordinatorModel = Objects.requireNonNull(coordinatorModel, "coordinatorModel");
        this.backend = Objects.requireNonNull(backend, "backend");
    }

    /**
     * Creates a causal-LM facade for a single local transformer executor.
     *
     * <p>The returned facade owns the supplied model and closes it from {@link #close()}.</p>
     */
    public static DefaultCausalLanguageModel local(AbstractModel model) {
        return local(model, SketchesSettings.DEFAULT);
    }

    public static DefaultCausalLanguageModel local(AbstractModel model, SketchesSettings sketchesSettings) {
        return new DefaultCausalLanguageModel(model, new LocalGenerationBackend(model), sketchesSettings);
    }

    /**
     * Generates text by running the shared generation loop against this model's backend.
     *
     * <p>The session id is forwarded to the backend so distributed implementations can associate rank-local KV state with
     * this request. The prompt must already be rendered into the text expected by the model, usually through
     * {@link #promptSupport()}.</p>
     */
    @Override
    public Response generate(UUID sessionId, PromptContext promptContext, GeneratorParameters generatorParameters,
            GenerateEvent eventFired) {
        if (generatorParameters.timeout.isPresent()) {
            TimeLimiter limiter = TimeLimiter.of(generatorParameters.timeout.get());
            try {
                return limiter.executeFutureSupplier(() -> CompletableFuture.supplyAsync(
                        () -> generateWithoutTimeout(sessionId, promptContext, generatorParameters, eventFired)));
            } catch (TimeoutException e) {
                throw new GenerationException("Generation timed out after " + generatorParameters.timeout.get(), e);
            } catch (ExecutionException e) {
                Throwable cause = e.getCause();
                if (cause instanceof RuntimeException runtime) throw runtime;
                throw new GenerationException("Generation failed", cause);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new GenerationException("Generation interrupted", e);
            } catch (Exception e) {
                throw new GenerationException("Generation failed", e);
            }
        }
        return generateWithoutTimeout(sessionId, promptContext, generatorParameters, eventFired);
    }

    private Response generateWithoutTimeout(UUID sessionId, PromptContext promptContext,
            GeneratorParameters generatorParameters, GenerateEvent eventFired) {
        coordinatorModel.deadReckoningHeader(promptContext, generatorParameters);
        Response response;
        if (coordinatorModel.usesModelSpecificGeneration()) {
            response = coordinatorModel.generate(sessionId, promptContext, generatorParameters, eventFired);
        } else {
            response = new GenerationEngineRef().generate(coordinatorModel, backend, sessionId, promptContext,
                    generatorParameters, eventFired);
        }
        coordinatorModel.deadReckoningResponse(response);
        return response;
    }

    @Override
    public Config getConfig() {
        return coordinatorModel.getConfig();
    }

    @Override
    public PreTrainedTokenizer getTokenizer() {
        return coordinatorModel.getTokenizer();
    }

    @Override
    public Optional<PromptSupport> promptSupport() {
        return coordinatorModel.promptSupport();
    }

    @Override
    public ToolCallParser getToolCallParser() {
        return coordinatorModel.getToolCallParser();
    }

    /**
     * Delegates to {@link #coordinatorModel} directly rather than checking {@code backend
     * instanceof LocalGenerationBackend} first -- {@code coordinatorModel}'s own
     * {@code tensorParallelContext} correctly reflects this rank's TP state regardless of which
     * backend wraps it, so {@code AbstractModel.setActiveAdapter}'s existing TP guard already
     * enforces the step 4 plan's Section 7 non-goal at this layer too. See step 4 plan Section
     * 3.1's "implementation-time item to verify" for why either placement is correct.
     */
    @Override
    public void registerLoraAdapter(String adapterId, LoraAdapter adapter) {
        coordinatorModel.registerLoraAdapter(adapterId, adapter);
    }

    @Override
    public void registerLoraAdapter(String adapterId, LoraAdapterModelFetcher fetcher) {
        coordinatorModel.registerLoraAdapter(adapterId, fetcher);
    }

    @Override
    public void unregisterLoraAdapter(String adapterId) {
        coordinatorModel.unregisterLoraAdapter(adapterId);
    }

    @Override
    public void setActiveAdapter(String adapterId) {
        coordinatorModel.setActiveAdapter(adapterId);
    }

    @Override
    public void clearActiveAdapter() {
        coordinatorModel.clearActiveAdapter();
    }

    /**
     * Returns the local transformer executor used by this facade.
     *
     * <p>This is an implementation escape hatch for tests and migration code. New user-facing code should use the
     * {@link CausalLanguageModel} interface rather than depending on {@link AbstractModel}.</p>
     */
    public AbstractModel localTransformerModel() {
        return coordinatorModel;
    }

    /**
     * Closes the backend first, then the coordinator model.
     *
     * <p>Backend cleanup may release remote sessions, worker clients, or local KV resources. The coordinator is closed in
     * a {@code finally} block so model memory is still released if backend cleanup fails.</p>
     */
    @Override
    public void close() throws IOException {
        try {
            backend.close();
        } catch (Exception e) {
            throw new IOException(e);
        } finally {
            coordinatorModel.close();
        }
    }
}
