package io.teknek.reckoning.model.gemma4;

import io.dropwizard.metrics5.MetricRegistry;
import io.dropwizard.metrics5.Timer;
import io.teknek.reckoning.DType;
import io.teknek.reckoning.generator.*;
import io.teknek.reckoning.generator2.Gemma4CausalSelfAttention2;
import io.teknek.reckoning.generator2.MLPBlock2;
import io.teknek.reckoning.generator2.RmsNorm2;
import io.teknek.reckoning.generator2.SampleOutputRef;
import io.teknek.reckoning.generator2.TransformerBlock2;
import io.teknek.reckoning.grace.PreTrainedTokenizer;
import io.teknek.reckoning.math.FloatConversions;
import io.teknek.reckoning.math.WrappedForkJoinPool;
import io.teknek.reckoning.model.llama.LlamaModel;
import io.teknek.reckoning.model.tensorparallel.TensorParallelCollectives;
import io.teknek.reckoning.model.tensorparallel.TensorParallelContext;
import io.teknek.reckoning.safetensors.Config;
import io.teknek.reckoning.safetensors.WeightLoader;
import io.teknek.reckoning.tensor.KvBufferCacheSettings;
import io.teknek.reckoning.tensor.TensorShape;
import io.teknek.reckoning.tensor2.TensorRef;
import io.teknek.reckoning.tensor2.Scale;
import io.teknek.reckoning.tensor2.Lighter;
import io.teknek.reckoning.toolcallparser.ToolCallParser;
import io.teknek.reckoning.tensor.kv.AttentionPattern;
import io.teknek.reckoning.tensor.kv.KvCacheSession;
import io.teknek.reckoning.tensor.kv.KvReadView;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;


public class Gemma4Model extends LlamaModel {
    private volatile TensorRef perLayerEmbeddingWeightsRef;
    private volatile TensorRef perLayerModelProjectionWeightsRef;
    private volatile TensorRef perLayerProjectionNormWeightsRef;
    private volatile TensorRef embedTokenWeightsRef;
    private final Lighter v2Lighter;

    /**
     * Not supported: {@code loadTransformerBlockWeights()} uses the independent
     * {@code Gemma4CausalSelfAttention}/{@code VariableMLPBlock} classes, which share no code with
     * {@code CausalSelfAttention}/{@code MLPBlock} and have no LoRA hook wired in. Overrides
     * {@code LlamaModel}'s inherited {@code true} back to {@code false}. See step 4 plan Section 6.
     */
    @Override
    protected boolean supportsLoraHotSwap() {
        return false;
    }

    /** TensorRef-native constructor. */
    public Gemma4Model(
            InferenceType inferenceType, Config config, WeightLoader weights, PreTrainedTokenizer tokenizer,
            DType workingDType, DType workingQType, Optional<DType> modelQType,
            MetricRegistry metricRegistry, KvBufferCacheSettings kvBufferCacheSettings,
            ToolCallParser toolCallParser, WrappedForkJoinPool pool, TensorParallelContext tensorParallelContext,
            TensorParallelCollectives tensorParallelCollectives, Optional<DType> outputHeadQuantization,
            Lighter lighter
    ) {
        super(inferenceType, config, weights, tokenizer, workingDType, workingQType, modelQType,
                metricRegistry, kvBufferCacheSettings, toolCallParser, pool, tensorParallelContext,
                tensorParallelCollectives, outputHeadQuantization, lighter);
        configureGemma4KvWidths();
        this.v2Lighter = lighter;
    }

    private void configureGemma4KvWidths() {
        Gemma4Config cfg = gemma4Config();
        int[] widths = cfg.layerTypes.stream().mapToInt(cfg::getLayerKvLength).toArray();
        int tpSize = tensorParallelContext.size();
        for (int i = 0; i < widths.length; i++) {
            widths[i] /= tpSize;
        }
        kvCacheManager.setLayerKvLengths(widths);
        if (kvPrefixSnapshotCache != null) {
            kvPrefixSnapshotCache.setLayerKvLengths(widths);
        }
    }

    @Override
    protected boolean usesTensorRefExecution() {
        return v2Lighter != null;
    }

    @Override
    public boolean usesKvCache2Generation() {
        return v2Lighter != null;
    }

    /** Gemma4's chat template emits BOS; runtime prompt construction retains exactly one BOS token. */
    @Override
    protected boolean addBosToken() {
        return true;
    }

    @Override
    public TensorRef batchForwardRef(int[] tokenIds, int startPos,
            io.teknek.reckoning.tensor.kv.KvCacheSession kvSession,
            Optional<java.util.function.Consumer<List<TensorRef>>> tensorReducer) {
        if (v2Lighter == null || gemma4Config().hiddenSizePerLayerInput == null) {
            return super.batchForwardRef(tokenIds, startPos, kvSession, tensorReducer);
        }
        V2SharedKeyValues sharedKvStates = kvSession.getOrAttach(V2SharedKeyValues.class, V2SharedKeyValues::new);
        TensorRef hiddenStates = embedInput.batchInputsToEmbeddings(tokenIds, startPos);
        emitLayerDebug(-1, "embedding", hiddenStates);
        TensorRef[] perLayerInputs = computePerLayerInputsRef(tokenIds, hiddenStates);
        try {
            for (int i = 0; i < perLayerInputs.length; i++) emitLayerDebug(i, "ple_input", perLayerInputs[i]);
            for (int layer = 0; layer < transformerBlocks2.length; layer++) {
                TensorRef residual = hiddenStates;
                hiddenStates = transformerBlocks2[layer].forward(residual, perLayerInputs[layer], startPos, kvSession,
                        tensorReducer, ForwardPhase.PREFILL, tokenIds.length, tokenIds.length, null);
                residual.close();
                emitLayerDebug(layer, "layer_output", hiddenStates);
            }
            kvSession.advanceLength(startPos + tokenIds.length);
            return hiddenStates;
        } finally {
            for (TensorRef perLayerInput : perLayerInputs) if (perLayerInput != null) perLayerInput.close();
        }
    }

    @Override
    public TensorRef forwardRef(int tokenId, int pos,
            io.teknek.reckoning.tensor.kv.KvCacheSession kvSession,
            Optional<java.util.function.Consumer<List<TensorRef>>> tensorReducer) {
        if (v2Lighter == null || gemma4Config().hiddenSizePerLayerInput == null) {
            return super.forwardRef(tokenId, pos, kvSession, tensorReducer);
        }
        V2SharedKeyValues sharedKvStates = kvSession.getOrAttach(V2SharedKeyValues.class, V2SharedKeyValues::new);
        TensorRef hiddenStates = embedInput.inputTokenToEmbedding(tokenId, pos);
        TensorRef[] perLayerInputs = computePerLayerInputsRef(new int[] { tokenId }, hiddenStates);
        try {
            for (int layer = 0; layer < transformerBlocks2.length; layer++) {
                TensorRef residual = hiddenStates;
                hiddenStates = transformerBlocks2[layer].forward(residual, perLayerInputs[layer], pos, kvSession,
                        tensorReducer, ForwardPhase.DECODE, 1, 1, null);
                residual.close();
            }
            kvSession.advanceLength(pos + 1);
            return hiddenStates;
        } finally {
            for (TensorRef perLayerInput : perLayerInputs) if (perLayerInput != null) perLayerInput.close();
        }
    }

    public static final class V2SharedKeyValues implements AutoCloseable {
        private static final class StoredKeyValues implements AutoCloseable {
            private final TensorRef key;
            private final TensorRef value;
            private boolean closed;

            private StoredKeyValues(TensorRef key, TensorRef value) {
                this.key = key;
                this.value = value;
            }

            @Override
            public synchronized void close() {
                if (closed) return;
                closed = true;
                key.close();
                value.close();
            }
        }

        private final Map<Integer, StoredKeyValues> values = new HashMap<>();

        public void store(int source, int startPosition, TensorRef key, TensorRef value, Lighter lighter) {
            StoredKeyValues previous = values.get(source);
            int previousRows = previous == null ? 0 : previous.key.shape().first();
            if (startPosition > previousRows) {
                throw new IllegalStateException("Missing shared KV prefix for source " + source
                        + " at position " + startPosition + "; available rows=" + previousRows);
            }
            int resultRows = startPosition + key.shape().first();
            TensorRef keyCopy = lighter.allocate(key.dType(), TensorShape.of(resultRows, key.shape().last()));
            TensorRef valueCopy = lighter.allocate(value.dType(), TensorShape.of(resultRows, value.shape().last()));
            try {
                if (previous != null) {
                    lighter.copy(previous.key, 0, keyCopy, 0, startPosition * key.shape().last());
                    lighter.copy(previous.value, 0, valueCopy, 0, startPosition * value.shape().last());
                }
                int keyOffset = startPosition * key.shape().last();
                int valueOffset = startPosition * value.shape().last();
                lighter.copy(key, 0, keyCopy, keyOffset, key.shape().first() * key.shape().last());
                lighter.copy(value, 0, valueCopy, valueOffset, value.shape().first() * value.shape().last());
                values.put(source, new StoredKeyValues(keyCopy, valueCopy));
                if (previous != null) previous.close();
            } catch (RuntimeException | Error e) {
                keyCopy.close();
                valueCopy.close();
                throw e;
            }
        }

        public void restorePrefix(int source, KvCacheSession session, int prefixRows, Lighter lighter) {
            if (prefixRows == 0 || values.containsKey(source)) return;
            try (KvReadView view = session.readView(source, prefixRows, AttentionPattern.CAUSAL);
                 TensorRef firstKey = view.keyRowCopyRef(0);
                 TensorRef firstValue = view.valueRowCopyRef(0)) {
                TensorRef key = lighter.allocate(firstKey.dType(),
                        TensorShape.of(prefixRows, firstKey.shape().last()));
                TensorRef value = lighter.allocate(firstValue.dType(),
                        TensorShape.of(prefixRows, firstValue.shape().last()));
                try {
                    for (int row = 0; row < prefixRows; row++) {
                        try (TensorRef keyRow = view.keyRowCopyRef(row);
                             TensorRef valueRow = view.valueRowCopyRef(row)) {
                            lighter.copy(keyRow, 0, key, row * key.shape().last(), key.shape().last());
                            lighter.copy(valueRow, 0, value, row * value.shape().last(), value.shape().last());
                        }
                    }
                    values.put(source, new StoredKeyValues(key, value));
                } catch (RuntimeException | Error e) {
                    key.close();
                    value.close();
                    throw e;
                }
            }
        }

        public TensorRef get(int source, boolean key) {
            StoredKeyValues current = values.get(source);
            if (current == null) throw new IllegalStateException("Missing V2 shared KV source " + source);
            return key ? current.key : current.value;
        }

        void clear() {
            values.values().forEach(StoredKeyValues::close);
            values.clear();
        }

        @Override
        public void close() {
            clear();
        }
    }

    @Override
    protected TransformerBlock2[] loadTransformerBlockWeights2() {
        if (v2Lighter == null) {
            throw new IllegalStateException("Gemma4 V2 requires a Lighter");
        }
        Gemma4Config cfg = gemma4Config();
        DType qType = modelQType.orElse(modelDType);
        String root = resolveTextModelRoot();
        TransformerBlock2[] blocks = new TransformerBlock2[cfg.numberOfLayers];
        int[] layerKvLengths = new int[cfg.numberOfLayers];
        for (int i = 0; i < cfg.numberOfLayers; i++) {
            String base = root + "layers." + i + ".";
            String type = cfg.layerTypes.get(i);
            int source = cfg.getSharedKvSourceLayer(i);
            String projectionBase = source >= 0 ? root + "layers." + source + "." : base;
            String qName = base + "self_attn.q_proj.weight";
            String kName = projectionBase + "self_attn.k_proj.weight";
            String vName = projectionBase + "self_attn.v_proj.weight";
            String oName = base + "self_attn.o_proj.weight";
            TensorRef q = registerModelTensorRef(loadV2(qName, qType));
            boolean sharedConsumer = source >= 0;
            TensorRef k = sharedConsumer ? null : registerModelTensorRef(loadV2(kName, qType));
            boolean keyEqualsValue = !sharedConsumer && cfg.attentionKEqV && "full_attention".equals(type);
            TensorRef v = sharedConsumer ? null : keyEqualsValue ? k : registerModelTensorRef(loadV2(vName, qType));
            TensorRef o = registerModelTensorRef(loadV2(oName, qType));
            TensorRef qNorm = registerModelTensorRef(loadV2(base + "self_attn.q_norm.weight", qType));
            TensorRef kNorm = sharedConsumer ? null
                    : registerModelTensorRef(loadV2(projectionBase + "self_attn.k_norm.weight", qType));
            layerKvLengths[i] = cfg.getLayerKvLength(type);
            int headDim = cfg.getLayerHeadDim(type);
            Gemma4CausalSelfAttention2 attention = new Gemma4CausalSelfAttention2(this, i, q, qNorm, k, v, kNorm, o,
                    v2Lighter, metricRegistry, cfg.numberOfHeads, cfg.getLayerKeyValueProjectionHeads(type),
                    headDim, cfg.getLayerQueryProjectionLength(type), cfg.getLayerKvLength(type),
                    cfg.ropeInvFreqsByLayerType.get(type), qName, kName, vName, oName,
                    "sliding_attention".equals(type), cfg.useAllBidirectionalAttention(),
                    cfg.slidingWindow == null ? cfg.contextLength : cfg.slidingWindow,
                    source >= 0 ? source : i);
            String mlp = base + "mlp.";
            String gateName = mlp + "gate_proj.weight";
            MLPBlock2 feedForward = new MLPBlock2(this, registerModelTensorRef(loadV2(gateName, qType)),
                    registerModelTensorRef(loadV2(mlp + "up_proj.weight", qType)),
                    registerModelTensorRef(loadV2(mlp + "down_proj.weight", qType)), v2Lighter,
                    gateName, mlp + "up_proj.weight", mlp + "down_proj.weight", cfg.getLayerHiddenLength(i));
            io.teknek.reckoning.generator2.FeedForward2 v2FeedForward = feedForward;
            io.teknek.reckoning.generator2.FeedForward2 moeFeedForward = null;
            Optional<io.teknek.reckoning.generator2.LayerNorm2> moeDenseNorm = Optional.empty();
            Optional<io.teknek.reckoning.generator2.LayerNorm2> moeInputNorm = Optional.empty();
            Optional<io.teknek.reckoning.generator2.LayerNorm2> moeOutputNorm = Optional.empty();
            if (cfg.enableMoeBlock) {
                moeFeedForward = new Gemma4MoeFeedForward2(this, cfg,
                        registerModelTensorRef(loadV2(base + "router.proj.weight", qType)),
                        registerModelTensorRef(loadV2(base + "router.scale", qType)),
                        registerModelTensorRef(loadV2(base + "router.per_expert_scale", qType)),
                        registerModelTensorRef(loadV2(base + "experts.gate_up_proj", qType)),
                        registerModelTensorRef(loadV2(base + "experts.down_proj", qType)), v2Lighter);
                moeDenseNorm = Optional.of(new RmsNorm2(this, registerModelTensorRef(
                        loadV2(base + "post_feedforward_layernorm_1.weight", qType)), 0.0f));
                moeInputNorm = Optional.of(new RmsNorm2(this, registerModelTensorRef(
                        loadV2(base + "pre_feedforward_layernorm_2.weight", qType)), 0.0f));
                moeOutputNorm = Optional.of(new RmsNorm2(this, registerModelTensorRef(
                        loadV2(base + "post_feedforward_layernorm_2.weight", qType)), 0.0f));
            }
            TensorRef pleGate = null;
            TensorRef pleProjection = null;
            Optional<io.teknek.reckoning.generator2.LayerNorm2> pleNorm = Optional.empty();
            if (cfg.hiddenSizePerLayerInput != null && weights.isWeightPresent(base + "per_layer_input_gate.weight")) {
                pleGate = registerModelTensorRef(loadV2(base + "per_layer_input_gate.weight", qType));
                pleProjection = registerModelTensorRef(loadV2(base + "per_layer_projection.weight", qType));
                pleNorm = Optional.of(new RmsNorm2(this, registerModelTensorRef(
                        loadV2(base + "post_per_layer_input_norm.weight", qType)), 0.0f));
            }
            blocks[i] = new TransformerBlock2(this, i,
                    Optional.of(new RmsNorm2(this, registerModelTensorRef(loadV2(base + "input_layernorm.weight", qType)), 0.0f)),
                    attention,
                    Optional.of(new RmsNorm2(this, registerModelTensorRef(loadV2(base + "post_attention_layernorm.weight", qType)), 0.0f)),
                    Optional.of(new RmsNorm2(this, registerModelTensorRef(loadV2(base + "pre_feedforward_layernorm.weight", qType)), 0.0f)),
                    v2FeedForward,
                    Optional.of(new RmsNorm2(this, registerModelTensorRef(loadV2(base + "post_feedforward_layernorm.weight", qType)), 0.0f)),
                    Optional.empty(), loadScalar(base + "layer_scalar"), pleGate,
                    pleProjection, pleNorm, cfg.hiddenSizePerLayerInput == null ? 0 : cfg.hiddenSizePerLayerInput,
                    moeDenseNorm, moeInputNorm, moeOutputNorm, moeFeedForward);
        }
        return blocks;
    }

    @Override
    protected SampleOutputRef loadOutputWeightsRef() {
        DType qType = modelQType.orElse(modelDType);
        TensorRef norm = registerModelTensorRef(loadV2(resolveTextModelRoot() + "norm.weight", qType));
        String headName = weights.isWeightPresent(resolveTextOutputWeight("lm_head.weight"))
                ? resolveTextOutputWeight("lm_head.weight") : resolveTextModelWeight("embed_tokens.weight");
        DType outputDType = outputHeadQuantization.orElse(workingDType);
        TensorRef loadedHead = weights.loadRef(headName);
        TensorRef head;
        if (loadedHead.dType() == outputDType) {
            head = loadedHead;
        } else {
            head = lighter.reshape(loadedHead, outputDType);
            loadedHead.close();
        }
        TensorRef outputHead = registerModelTensorRef(head);
        return new SampleOutputRef() {
            public io.teknek.reckoning.generator2.LayerNorm2 outputLayerNorm() { return new RmsNorm2(Gemma4Model.this, norm, 0.0f); }
            public TensorRef outputLogitsWeights() { return outputHead; }
        };
    }

    private TensorRef loadV2(String name, DType qType) {
        TensorRef ref = weights.loadRef(name);
        if (ref.dType() != qType && lighter.shouldQuantizeForEfficiency(ref, qType)) {
            TensorRef converted = lighter.reshape(ref, qType);
            ref.close();
            return converted;
        }
        return ref;
    }

    @Override
    protected EmbedInput loadInputWeights() {
        return loadInputWeightsV2();
    }

    private EmbedInput loadInputWeightsV2() {
        TensorRef embeddingWeights = registerModelTensorRef(loadV2(resolveTextModelWeight("embed_tokens.weight"),
                workingDType));
        final float embeddingScalingFactor = FloatConversions.bFloat16ToFloat32(
                FloatConversions.float32ToBFloat16((float) Math.pow(config.embeddingLength, 0.5)));
        return new EmbedInput(this) {
            @Override
            public TensorRef inputTokenToEmbedding(int inputToken, int position) {
                try (TensorRef row = embeddingWeights.slice(inputToken)) {
                    TensorRef embedding = makeDenseTensorRef(1, config.embeddingLength);
                    try {
                        if (row.dType() != embedding.dType()) {
                            try (TensorRef converted = v2Lighter.reshape(row, embedding.dType())) {
                                v2Lighter.copy(converted, 0, embedding, 0, config.embeddingLength);
                            }
                        } else {
                            v2Lighter.copy(row, 0, embedding, 0, config.embeddingLength);
                        }
                        v2Lighter.scale(new Scale(embeddingScalingFactor).target(embedding)
                                .offsetAndLength(0, config.embeddingLength));
                        return embedding;
                    } catch (RuntimeException | Error e) {
                        embedding.close();
                        throw e;
                    }
                }
            }
        };
    }

    @Override
    protected Response postProcessResponse(Response response) {
        Gemma4ResponseParser.Parsed parsed = Gemma4ResponseParser.parse(
                response.responseTextWithSpecialTokens,
                response.responseText
        );
        return response.copyWithText(parsed.content(), response.responseTextWithSpecialTokens, parsed.reasoning());
    }

    private synchronized void ensurePerLayerWeightsLoadedRef() {
        Gemma4Config cfg = gemma4Config();
        if (cfg.hiddenSizePerLayerInput == null || perLayerModelProjectionWeightsRef != null) {
            return;
        }
        DType qType = modelQType.orElse(modelDType);
        perLayerModelProjectionWeightsRef = registerModelTensorRef(loadV2(
                resolveTextModelWeight("per_layer_model_projection.weight"), qType));
        perLayerProjectionNormWeightsRef = registerModelTensorRef(loadV2(
                resolveTextModelWeight("per_layer_projection_norm.weight"), qType));
    }

    private TensorRef[] computePerLayerInputsRef(int[] tokenIds, TensorRef embeddings) {
        Gemma4Config cfg = gemma4Config();
        if (cfg.hiddenSizePerLayerInput == null) {
            return null;
        }
        ensurePerLayerWeightsLoadedRef();
        int batchSize = tokenIds.length;
        int packedLength = cfg.numberOfLayers * cfg.hiddenSizePerLayerInput;
        TensorRef tokenIdentity = makeDenseTensorRef(batchSize, packedLength);
        TensorRef perLayerProjection = makeDenseTensorRef(batchSize, packedLength);
        try {
            loadPerLayerIdentityRows(tokenIds, tokenIdentity);
            getCompositeOps().multiplyInPlace(new io.teknek.reckoning.tensor2.MultiplyInPlace(
                    perLayerEmbeddingScale()).target(tokenIdentity).offsetAndLength(0, packedLength));
            emitLayerDebug(-1, "ple_token_identity", tokenIdentity);
            getLighter().batchDotProduct(new io.teknek.reckoning.tensor2.BatchDotProduct()
                    .result(perLayerProjection).a(embeddings).b(perLayerModelProjectionWeightsRef)
                    .aColumnOffset(0).bColumnOffset(0).columnLength(cfg.embeddingLength)
                    .resultRowOffset(0).bRowOffset(0).rowChunkSize(packedLength));
            getCompositeOps().multiplyInPlace(new io.teknek.reckoning.tensor2.MultiplyInPlace(
                    perLayerModelProjectionScale()).target(perLayerProjection).offsetAndLength(0, packedLength));
            emitLayerDebug(-1, "ple_projection_scaled", perLayerProjection);
            getCompositeOps().groupedRmsNormInPlace(perLayerProjection, cfg.numberOfLayers,
                    cfg.hiddenSizePerLayerInput, cfg.layerNormEps, perLayerProjectionNormWeightsRef);
            emitLayerDebug(-1, "ple_projection_norm", perLayerProjection);
            getLighter().accumulate(new io.teknek.reckoning.tensor2.Accumulate(tokenIdentity)
                    .into(perLayerProjection).offsetAndLength(0, packedLength));
            getCompositeOps().multiplyInPlace(new io.teknek.reckoning.tensor2.MultiplyInPlace(
                    perLayerInputScale()).target(perLayerProjection).offsetAndLength(0, packedLength));
            emitLayerDebug(-1, "ple_combined", perLayerProjection);
            tokenIdentity.close();
            tokenIdentity = null;
            TensorRef[] split = splitPerLayerInputsRef(perLayerProjection, batchSize, cfg.numberOfLayers,
                    cfg.hiddenSizePerLayerInput);
            perLayerProjection.close();
            perLayerProjection = null;
            return split;
        } finally {
            if (tokenIdentity != null) tokenIdentity.close();
            if (perLayerProjection != null) perLayerProjection.close();
        }
    }

    private void loadPerLayerIdentityRows(int[] tokenIds, TensorRef destination) {
        String name = resolveTextModelWeight("embed_tokens_per_layer.weight");
        for (int row = 0; row < tokenIds.length; row++) {
            try (TensorRef source = weights.loadRows(name, tokenIds[row], 1)) {
                if (source.dType() == destination.dType()) {
                    getLighter().copy(source, 0, destination, destination.shape().getOffset(row, 0),
                            destination.shape().last());
                } else {
                    try (TensorRef converted = lighter.reshape(source, destination.dType())) {
                        getLighter().copy(converted, 0, destination,
                                destination.shape().getOffset(row, 0), destination.shape().last());
                    }
                }
            }
        }
    }

    private TensorRef[] splitPerLayerInputsRef(TensorRef packed, int batchSize, int numberOfLayers,
            int hiddenSizePerLayerInput) {
        TensorRef[] split = new TensorRef[numberOfLayers];
        try {
            for (int layer = 0; layer < numberOfLayers; layer++) {
                TensorRef output = makeDenseTensorRef(batchSize, hiddenSizePerLayerInput);
                split[layer] = output;
                int sourceOffset = layer * hiddenSizePerLayerInput;
                for (int row = 0; row < batchSize; row++) {
                    getLighter().copy(packed, packed.shape().getOffset(row, sourceOffset), output,
                            output.shape().getOffset(row, 0), hiddenSizePerLayerInput);
                }
            }
            return split;
        } catch (RuntimeException | Error e) {
            for (TensorRef value : split) if (value != null) value.close();
            throw e;
        }
    }

    private float loadScalar(String name) {
        if (!weights.isWeightPresent(name)) {
            return 1.0f;
        }
        try (TensorRef tensor = weights.loadRef(name)) {
            return tensor.shape().dims() == 1 ? tensor.get(0) : tensor.get(0, 0);
        }
    }

    private Gemma4Config gemma4Config() {
        return (Gemma4Config) config;
    }

    private float perLayerEmbeddingScale() {
        Gemma4Config cfg = gemma4Config();
        return cfg.hiddenSizePerLayerInput == null ? 1.0f : (float) Math.sqrt(cfg.hiddenSizePerLayerInput);
    }

    private float perLayerInputScale() {
        return (float) Math.pow(2.0, -0.5);
    }

    private float perLayerModelProjectionScale() {
        return (float) Math.pow(config.embeddingLength, -0.5);
    }

    private String resolveTextModelRoot() {
        List<String> candidates = List.of("model.language_model.", "language_model.model.", "language_model.", "model.");
        for (String candidate : candidates) {
            if (weights.isWeightPresent(candidate + "embed_tokens.weight")
                    || weights.isWeightPresent(candidate + "layers.0.self_attn.q_proj.weight")) {
                return candidate;
            }
        }
        throw new IllegalArgumentException("Unable to locate Gemma4 text model root");
    }

    private String resolveTextModelWeight(String suffix) {
        String root = resolveTextModelRoot();
        String name = root + suffix;
        if (weights.isWeightPresent(name) || weights.isWeightPresent(name + "-part-0")) {
            return name;
        }
        throw new IllegalArgumentException("Missing Gemma4 text weight " + name);
    }

    private String resolveTextOutputWeight(String suffix) {
        List<String> candidates = List.of(suffix, "language_model." + suffix, resolveTextModelRoot() + suffix);
        for (String candidate : candidates) {
            if (weights.isWeightPresent(candidate)) {
                return candidate;
            }
        }
        return suffix;
    }

}
