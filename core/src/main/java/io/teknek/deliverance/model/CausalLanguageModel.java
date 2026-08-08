package io.teknek.deliverance.model;

import io.teknek.deliverance.generator.Generator;
import io.teknek.deliverance.grace.PreTrainedTokenizer;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.safetensors.LoraAdapter;
import io.teknek.deliverance.safetensors.fetch.LoraAdapterModelFetcher;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import io.teknek.deliverance.toolcallparser.ToolCallParser;

import java.util.Optional;

/**
 * Text-generation model loaded for causal language-model inference.
 *
 * <p>Create instances with {@link AutoModelForCausaLm#fromPretrained(io.teknek.deliverance.safetensors.fetch.ModelFetcher)}
 * or {@link AutoModelForCausaLm#newBuilder(io.teknek.deliverance.safetensors.fetch.ModelFetcher)}. Use
 * {@link #promptSupport()} to build model-specific chat prompts, then call {@link #generate(java.util.UUID,
 * io.teknek.deliverance.safetensors.prompt.PromptContext, io.teknek.deliverance.generator.GeneratorParameters,
 * GenerateEvent)} to produce tokens.</p>
 *
 * <p>The interface is intentionally the same for local and tensor-parallel inference. Backend choice can affect
 * performance, memory use, cache behavior, and floating-point results. In particular, current tensor-parallel generation
 * does not expose the local prefix-cache reuse path through this interface; rank-local KV state is request scoped. Treat
 * tensor-parallel output equivalence as model/provider-specific rather than guaranteed.</p>
 *
 * <pre>{@code
 * try (CausalLanguageModel model = AutoModelForCausaLm.fromPretrained(fetcher)) {
 *     PromptContext prompt = model.promptSupport().orElseThrow()
 *             .builder()
 *             .addUserMessage("Explain tensor parallelism briefly.")
 *             .build();
 *
 *     Response response = model.generate(
 *             UUID.randomUUID(),
 *             prompt,
 *             new GeneratorParameters().withMaxTokens(128),
 *             new DoNothingGenerateEvent());
 * }
 * }</pre>
 */
public interface CausalLanguageModel extends Generator {
    /**
     * Returns the loaded model configuration used for inference limits and model metadata.
     *
     * <p>The returned configuration is the runtime configuration for this loaded model. Callers commonly use it to
     * inspect context length, vocabulary size, architecture dimensions, or special token ids.</p>
     */
    Config getConfig();

    /**
     * Returns the tokenizer paired with this model.
     *
     * <p>Most chat callers should prefer {@link #promptSupport()} for model-specific prompt construction. Direct
     * tokenizer access is useful for diagnostics, token counting, and advanced prompt handling.</p>
     */
    PreTrainedTokenizer getTokenizer();

    /**
     * Returns prompt-template support when the loaded model provides a chat or instruction template.
     *
     * <p>If empty, callers can still pass a raw {@code PromptContext} to {@link #generate}, but Deliverance cannot build
     * model-specific chat prompts automatically.</p>
     */
    Optional<PromptSupport> promptSupport();

    /**
     * Returns the parser used to interpret generated tool-call text for this model family.
     */
    ToolCallParser getToolCallParser();

    /**
     * Registers a LoRA adapter for runtime hot-swap under {@code adapterId}, without activating
     * it. Call {@link #setActiveAdapter(String)} to actually apply it during generation.
     *
     * <p>Only supported by local, non-tensor-parallel backends over an opted-in model family --
     * see the step 4 plan (LoRA runtime hot-swap), Sections 6 and 7. Registration, activation, and
     * clearing are equally not thread-safe against concurrent {@link #generate} calls and must be
     * externally serialized by the caller, exactly like {@code generate} itself.</p>
     */
    void registerLoraAdapter(String adapterId, LoraAdapter adapter);

    /** Convenience overload that fetches and loads the adapter before registering it. */
    void registerLoraAdapter(String adapterId, LoraAdapterModelFetcher fetcher);

    /** Unregisters and closes a previously-registered adapter. It must not be the active adapter. */
    void unregisterLoraAdapter(String adapterId);

    /** Activates a previously-registered LoRA adapter; every subsequent forward pass applies its deltas. */
    void setActiveAdapter(String adapterId);

    /** Deactivates the currently active LoRA adapter, if any, restoring plain base-model behavior. */
    void clearActiveAdapter();
}
