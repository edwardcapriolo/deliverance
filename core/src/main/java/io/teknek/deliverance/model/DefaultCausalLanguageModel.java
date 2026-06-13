package io.teknek.deliverance.model;

import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.grace.PreTrainedTokenizer;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import io.teknek.deliverance.toolcallparser.ToolCallParser;

import java.io.IOException;
import java.util.Objects;
import java.util.Optional;
import java.util.UUID;

/**
 * Standard {@link CausalLanguageModel} returned by the causal-LM loader.
 *
 * <p>The facade exposes the task-level API while delegating transformer execution to a {@link GenerationBackend}. The
 * same facade can therefore serve a normal local model or a distributed/tensor-parallel runtime without changing caller
 * code. The coordinator model supplies tokenizer, prompt support, output projection, sampling helpers, and
 * model-specific response post-processing.</p>
 */
public final class DefaultCausalLanguageModel implements CausalLanguageModel {
    private final AbstractModel coordinatorModel;
    private final GenerationBackend backend;
    private final GenerationEngine engine = new GenerationEngine();

    public DefaultCausalLanguageModel(AbstractModel coordinatorModel, GenerationBackend backend) {
        this.coordinatorModel = Objects.requireNonNull(coordinatorModel, "coordinatorModel");
        this.backend = Objects.requireNonNull(backend, "backend");
    }

    /**
     * Creates a causal-LM facade for a single local transformer executor.
     *
     * <p>The returned facade owns the supplied model and closes it from {@link #close()}.</p>
     */
    public static DefaultCausalLanguageModel local(AbstractModel model) {
        return new DefaultCausalLanguageModel(model, new LocalGenerationBackend(model));
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
        return engine.generate(coordinatorModel, backend, sessionId, promptContext, generatorParameters, eventFired);
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
