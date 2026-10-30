package io.teknek.deliverance.generator2;

import io.dropwizard.metrics5.Timer;
import io.teknek.deliverance.generator.ForwardPhase;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.kv.KvCacheSession;
import io.teknek.deliverance.tensor2.CompositeOps;
import io.teknek.deliverance.tensor2.TensorRef;
import io.teknek.deliverance.tensor2.MultiplyInPlace;
import io.teknek.deliverance.math.ActivationFunction;

import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;

/** TensorRef-native transformer block. */
public class TransformerBlock2 {
    private final AbstractModel model;
    final int layerIndex;
    final Optional<LayerNorm2> preAttentionNorm;
    final SelfAttention2 attention;
    final Optional<LayerNorm2> postAttentionNorm;
    final Optional<LayerNorm2> preFFNorm;
    final FeedForward2 ffBlock;
    final Optional<LayerNorm2> postFFNorm;
    final Optional<LayerNorm2> preResponseNorm;
    final ConfigurableTensorProvider configurableTensorProvider;
    private final CompositeOps compositeOps;
    private final float layerScalar;
    private final TensorRef pleGateWeights;
    private final TensorRef pleProjectionWeights;
    private final Optional<LayerNorm2> pleNorm;
    private final int pleLength;
    private final Optional<LayerNorm2> moeDenseNorm;
    private final Optional<LayerNorm2> moeInputNorm;
    private final Optional<LayerNorm2> moeOutputNorm;
    private final FeedForward2 moeBlock;

    public TransformerBlock2(AbstractModel model, int layerIndex, LayerNorm2 preAttentionNorm,
            Qwen3KvCacheSelfAttention2 attention, LayerNorm2 postAttentionNorm, FeedForward2 mlp) {
        this(model, layerIndex, Optional.of(preAttentionNorm), attention, Optional.empty(),
                Optional.of(postAttentionNorm), mlp, Optional.empty(), Optional.empty(),
                model.getConfigurableTensorProvider(), 1.0f);
    }

    public TransformerBlock2(
            AbstractModel model,
            int layerIndex,
            Optional<LayerNorm2> preAttentionNorm,
            SelfAttention2 attention,
            Optional<LayerNorm2> postAttentionNorm,
            Optional<LayerNorm2> preFFNorm,
            FeedForward2 ffBlock,
            Optional<LayerNorm2> postFFNorm,
            Optional<LayerNorm2> preResponseNorm,
            ConfigurableTensorProvider configurableTensorProvider
    ) {
        this(model, layerIndex, preAttentionNorm, attention, postAttentionNorm, preFFNorm, ffBlock,
                postFFNorm, preResponseNorm, configurableTensorProvider, 1.0f);
    }

    public TransformerBlock2(
            AbstractModel model, int layerIndex, Optional<LayerNorm2> preAttentionNorm, SelfAttention2 attention,
            Optional<LayerNorm2> postAttentionNorm, Optional<LayerNorm2> preFFNorm, FeedForward2 ffBlock,
            Optional<LayerNorm2> postFFNorm, Optional<LayerNorm2> preResponseNorm,
            ConfigurableTensorProvider configurableTensorProvider, float layerScalar
    ) {
        this(model, layerIndex, preAttentionNorm, attention, postAttentionNorm, preFFNorm, ffBlock, postFFNorm,
                preResponseNorm, configurableTensorProvider, layerScalar, null, null, Optional.empty(), 0);
    }

    public TransformerBlock2(AbstractModel model, int layerIndex, Optional<LayerNorm2> preAttentionNorm,
            SelfAttention2 attention, Optional<LayerNorm2> postAttentionNorm, Optional<LayerNorm2> preFFNorm,
            FeedForward2 ffBlock, Optional<LayerNorm2> postFFNorm, Optional<LayerNorm2> preResponseNorm,
            ConfigurableTensorProvider configurableTensorProvider, float layerScalar, TensorRef pleGateWeights,
            TensorRef pleProjectionWeights, Optional<LayerNorm2> pleNorm, int pleLength) {
        this(model, layerIndex, preAttentionNorm, attention, postAttentionNorm, preFFNorm, ffBlock, postFFNorm,
                preResponseNorm, configurableTensorProvider, layerScalar, pleGateWeights, pleProjectionWeights,
                pleNorm, pleLength, Optional.empty(), Optional.empty(), Optional.empty(), null);
    }

    public TransformerBlock2(AbstractModel model, int layerIndex, Optional<LayerNorm2> preAttentionNorm,
            SelfAttention2 attention, Optional<LayerNorm2> postAttentionNorm, Optional<LayerNorm2> preFFNorm,
            FeedForward2 ffBlock, Optional<LayerNorm2> postFFNorm, Optional<LayerNorm2> preResponseNorm,
            ConfigurableTensorProvider configurableTensorProvider, float layerScalar, TensorRef pleGateWeights,
            TensorRef pleProjectionWeights, Optional<LayerNorm2> pleNorm, int pleLength,
            Optional<LayerNorm2> moeDenseNorm, Optional<LayerNorm2> moeInputNorm,
            Optional<LayerNorm2> moeOutputNorm, FeedForward2 moeBlock) {
        this.model = java.util.Objects.requireNonNull(model, "model");
        this.layerIndex = layerIndex;
        this.preAttentionNorm = java.util.Objects.requireNonNull(preAttentionNorm, "preAttentionNorm");
        this.attention = java.util.Objects.requireNonNull(attention, "attention");
        this.postAttentionNorm = java.util.Objects.requireNonNull(postAttentionNorm, "postAttentionNorm");
        this.preFFNorm = java.util.Objects.requireNonNull(preFFNorm, "preFFNorm");
        this.ffBlock = java.util.Objects.requireNonNull(ffBlock, "ffBlock");
        this.postFFNorm = java.util.Objects.requireNonNull(postFFNorm, "postFFNorm");
        this.preResponseNorm = java.util.Objects.requireNonNull(preResponseNorm, "preResponseNorm");
        this.configurableTensorProvider = configurableTensorProvider;
        this.compositeOps = new CompositeOps(model.getLighter(), model.getMetricRegistry());
        this.layerScalar = layerScalar;
        this.pleGateWeights = pleGateWeights;
        this.pleProjectionWeights = pleProjectionWeights;
        this.pleNorm = pleNorm;
        this.pleLength = pleLength;
        this.moeDenseNorm = moeDenseNorm;
        this.moeInputNorm = moeInputNorm;
        this.moeOutputNorm = moeOutputNorm;
        this.moeBlock = moeBlock;
    }

    public TensorRef forward(TensorRef embedding, int startPosition, KvCacheSession kvSession,
            Optional<Consumer<List<TensorRef>>> tensorReducer, ForwardPhase phase) {
        return forward(embedding, startPosition, kvSession, tensorReducer, phase,
                1, (int) embedding.shape().first(), null);
    }

    public TensorRef forward(TensorRef embedding, int startPosition, KvCacheSession kvSession,
            Optional<Consumer<List<TensorRef>>> tensorReducer, ForwardPhase phase, int batchSize,
            int sequenceLength, int[] attentionMask) {
        return forward(embedding, null, startPosition, kvSession, tensorReducer, phase, batchSize, sequenceLength,
                attentionMask);
    }

    public TensorRef forward(TensorRef embedding, TensorRef perLayerInput, int startPosition,
            KvCacheSession kvSession, Optional<Consumer<List<TensorRef>>> tensorReducer, ForwardPhase phase,
            int batchSize, int sequenceLength, int[] attentionMask) {
        Timer timer = InferenceProfiler.timer(model.getMetricRegistry(), "transformerblock.forward");
        try (Timer.Context ignored = timer.time()) {
            TensorRef residual = embedding;
            TensorRef preAttentionHiddenStates = preAttentionNorm.map(ln -> ln.forward(embedding)).orElse(embedding);
            TensorRef postAttention;
            TensorRef hiddenStates;
            try (AbstractModel.TensorRefLease preAttentionInput = preAttentionProjectionInput(preAttentionHiddenStates)) {
                postAttention = attention.forward(preAttentionInput.tensor(), startPosition, kvSession, tensorReducer, phase,
                        batchSize, sequenceLength, attentionMask);
            }
            hiddenStates = maybeApplyNorm(postAttention, postAttentionNorm);
            model.emitLayerDebug(layerIndex, "post_attention_norm", hiddenStates);
            applyResidual(hiddenStates, residual, "post_attention_residual");
            model.emitLayerDebug(layerIndex, "post_attention_residual", hiddenStates);

            TensorRef preFeedForward = preFFNorm.map(ln -> ln.forward(hiddenStates)).orElse(hiddenStates);
            model.emitLayerDebug(layerIndex, "pre_ff_norm", preFeedForward);
            TensorRef postFeedForward;
            try (AbstractModel.TensorRefLease preFeedForwardInput = preFFProjectionInput(preFeedForward)) {
                postFeedForward = ffBlock.forward(preFeedForwardInput.tensor(), tensorReducer, phase);
            }
            model.emitLayerDebug(layerIndex, "post_ff_output", postFeedForward);

            TensorRef postFeedForwardNorm = moeBlock != null && moeDenseNorm.isPresent()
                    && moeInputNorm.isPresent() && moeOutputNorm.isPresent()
                    ? applyMoeBranch(postFeedForward, hiddenStates, tensorReducer, phase)
                    : maybeApplyNorm(postFeedForward, postFFNorm);

            applyResidual(postFeedForwardNorm, hiddenStates, "post_ff_residual");
            model.emitLayerDebug(layerIndex, "post_ff_residual", postFeedForwardNorm);
            postFeedForwardNorm = applyPle(postFeedForwardNorm, perLayerInput, batchSize);

            postFeedForwardNorm = applyLayerScalar(postFeedForwardNorm);
            model.emitLayerDebug(layerIndex, "layer_scalar", postFeedForwardNorm);

            if (preAttentionHiddenStates != embedding) preAttentionHiddenStates.close();
            if (preFeedForward != hiddenStates) {
                hiddenStates.close();
                preFeedForward.close();
            } else {
                hiddenStates.close();
            }

            return maybeApplyNorm(postFeedForwardNorm, preResponseNorm);
        }
    }

    private AbstractModel.TensorRefLease preAttentionProjectionInput(TensorRef tensor) {
        return model.maybeQuantizeReadOnly(tensor, "transformerblock.maybe_quantize.pre_attention");
    }

    private AbstractModel.TensorRefLease preFFProjectionInput(TensorRef tensor) {
        return model.maybeQuantizeReadOnly(tensor, "transformerblock.maybe_quantize.pre_ff");
    }

    private TensorRef maybeApplyNorm(TensorRef tensor, Optional<LayerNorm2> norm) {
        return norm.map(ln -> {
            TensorRef output = ln.forward(tensor);
            tensor.close();
            return output;
        }).orElse(tensor);
    }

    private TensorRef applyMoeBranch(TensorRef denseOutput, TensorRef residual,
            Optional<Consumer<List<TensorRef>>> tensorReducer, ForwardPhase phase) {
        TensorRef denseBranch = moeDenseNorm.get().forward(denseOutput);
        TensorRef expertInput = moeInputNorm.get().forward(residual);
        TensorRef expertOutput;
        try {
            expertOutput = moeBlock.forward(expertInput, tensorReducer, phase);
        } finally {
            expertInput.close();
        }
        TensorRef expertBranch = moeOutputNorm.get().forward(expertOutput);
        expertOutput.close();
        compositeOps.addResidual(denseBranch, expertBranch, 1.0f);
        expertBranch.close();
        return maybeApplyNorm(denseBranch, postFFNorm);
    }

    private TensorRef applyPle(TensorRef hiddenStates, TensorRef perLayerInput, int batchSize) {
        if (perLayerInput == null || pleGateWeights == null || pleProjectionWeights == null || pleNorm.isEmpty()) {
            return hiddenStates;
        }
        TensorRef pleGate = model.makeDenseTensorRef(batchSize, pleLength);
        try {
            model.getLighter().dotProductRows(pleGate, hiddenStates, pleGateWeights, 0, 0,
                    model.getConfig().embeddingLength, 0, pleLength, 0);
            model.emitLayerDebug(layerIndex, "ple_gate", pleGate);
            TensorRef pleGated = model.makeDenseTensorRef(batchSize, pleLength);
            try {
                for (int row = 0; row < batchSize; row++) {
                    for (int column = 0; column < pleLength; column++) {
                        pleGated.set(ActivationFunction.eval(model.getConfig().activationFunction,
                                pleGate.get(row, column)) * perLayerInput.get(row, column), row, column);
                    }
                }
                model.emitLayerDebug(layerIndex, "ple_gated", pleGated);
                TensorRef pleProjected = model.makeDenseTensorRef(batchSize, model.getConfig().embeddingLength);
                try (AbstractModel.TensorRefLease gatedQ = model.maybeQuantizeReadOnly(pleGated,
                        "gemma4transformerblock2.maybe_quantize.per_layer_input")) {
                    model.getLighter().dotProductRows(pleProjected, gatedQ.tensor(), pleProjectionWeights, 0, 0,
                            pleLength, 0, model.getConfig().embeddingLength, 0);
                }
                model.emitLayerDebug(layerIndex, "ple_projected", pleProjected);
                TensorRef pleNormalized = pleNorm.get().forward(pleProjected);
                model.emitLayerDebug(layerIndex, "ple_normalized", pleNormalized);
                pleProjected.close();
                compositeOps.addResidual(pleNormalized, hiddenStates, 1.0f);
                hiddenStates.close();
                model.emitLayerDebug(layerIndex, "post_ple", pleNormalized);
                return pleNormalized;
            } finally {
                pleGated.close();
            }
        } finally {
            pleGate.close();
        }
    }

    private void applyResidual(TensorRef target, TensorRef residual, String name) {
        compositeOps.addResidual(target, residual, model.getConfig().residualMultiplier == null
                ? 1.0f : model.getConfig().residualMultiplier);
    }

    private TensorRef applyLayerScalar(TensorRef hiddenStates) {
        if (layerScalar != 1.0f) {
            compositeOps.multiplyInPlace(new MultiplyInPlace(layerScalar)
                    .target(hiddenStates).offsetAndLength(0, model.getConfig().embeddingLength));
        }
        return hiddenStates;
    }

    private void emit(String stage, TensorRef tensor) {
        model.emitLayerDebug(layerIndex, stage, tensor);
    }
}
