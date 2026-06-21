package io.teknek.deliverance.generator;

import com.codahale.metrics.Timer;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.KvBufferCache;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.function.Consumer;

import net.jafama.FastMath;

public class Gemma4TransformerBlock extends TransformerBlock {
    private static final Logger logger = LoggerFactory.getLogger(Gemma4TransformerBlock.class);
    private static final boolean DEBUG_BLOCK_SUMMARIES = false;
    private final AbstractModel model;
    private final ConfigurableTensorProvider configurableTensorProvider;
    private final Optional<AbstractTensor> perLayerInputGateWeights;
    private final Optional<AbstractTensor> perLayerProjectionWeights;
    private final Optional<LayerNorm> postPerLayerInputNorm;
    private final int perLayerInputLength;
    private final float layerScalar;
    private final ActivationFunction.Type activationFunction;
    private final Optional<LayerNorm> postFeedForwardNorm1;
    private final Optional<LayerNorm> preFeedForwardNorm2;
    private final Optional<LayerNorm> postFeedForwardNorm2;
    private final Optional<AbstractTensor> routerProjectionWeights;
    private final Optional<AbstractTensor> routerScaleWeights;
    private final Optional<AbstractTensor> routerPerExpertScaleWeights;
    private final Optional<AbstractTensor> expertGateUpWeights;
    private final Optional<AbstractTensor> expertDownWeights;
    private final int numberOfExperts;
    private final int topKExperts;
    private final int moeIntermediateLength;

    public Gemma4TransformerBlock(
            AbstractModel model,
            int layerIndex,
            LayerNorm preAttentionNorm,
            SelfAttention attention,
            LayerNorm postAttentionNorm,
            LayerNorm preFFNorm,
            FeedForward ffBlock,
            LayerNorm postFFNorm,
            ConfigurableTensorProvider configurableTensorProvider,
            Optional<AbstractTensor> perLayerInputGateWeights,
            Optional<AbstractTensor> perLayerProjectionWeights,
            Optional<LayerNorm> postPerLayerInputNorm,
            int perLayerInputLength,
            float layerScalar,
            ActivationFunction.Type activationFunction,
            Optional<LayerNorm> postFeedForwardNorm1,
            Optional<LayerNorm> preFeedForwardNorm2,
            Optional<LayerNorm> postFeedForwardNorm2,
            Optional<AbstractTensor> routerProjectionWeights,
            Optional<AbstractTensor> routerScaleWeights,
            Optional<AbstractTensor> routerPerExpertScaleWeights,
            Optional<AbstractTensor> expertGateUpWeights,
            Optional<AbstractTensor> expertDownWeights,
            int numberOfExperts,
            int topKExperts,
            int moeIntermediateLength
    ) {
        super(model, layerIndex, preAttentionNorm, attention, postAttentionNorm, preFFNorm, ffBlock, postFFNorm,
                configurableTensorProvider);
        this.model = model;
        this.configurableTensorProvider = configurableTensorProvider;
        this.perLayerInputGateWeights = perLayerInputGateWeights;
        this.perLayerProjectionWeights = perLayerProjectionWeights;
        this.postPerLayerInputNorm = postPerLayerInputNorm;
        this.perLayerInputLength = perLayerInputLength;
        this.layerScalar = layerScalar;
        this.activationFunction = activationFunction;
        this.postFeedForwardNorm1 = postFeedForwardNorm1;
        this.preFeedForwardNorm2 = preFeedForwardNorm2;
        this.postFeedForwardNorm2 = postFeedForwardNorm2;
        this.routerProjectionWeights = routerProjectionWeights;
        this.routerScaleWeights = routerScaleWeights;
        this.routerPerExpertScaleWeights = routerPerExpertScaleWeights;
        this.expertGateUpWeights = expertGateUpWeights;
        this.expertDownWeights = expertDownWeights;
        this.numberOfExperts = numberOfExperts;
        this.topKExperts = topKExperts;
        this.moeIntermediateLength = moeIntermediateLength;
    }

    public AbstractTensor forward(
            AbstractTensor embedding,
            AbstractTensor perLayerInput,
            int position,
            KvBufferCache.KvBuffer kvBuffer,
            Optional<Consumer<List<AbstractTensor>>> tensorReducer
    ) {
        Timer.Context totalTimer = model.getMetricRegistry().timer("gemma4.block." + layerIndex + ".total").time();
        try {
        AbstractTensor lnemb = preAttentionNorm.map(ln -> ln.forward(embedding)).orElse(embedding);
        logIfInteresting("pre_attn_norm", lnemb);
        AbstractTensor postAttention;
        try (AbstractTensor qlnemb = model.maybeQuantize(lnemb)) {
            postAttention = attention.forward(qlnemb, position, kvBuffer, tensorReducer);
        }
        AbstractTensor lnattn = maybeApplyNorm(postAttention, postAttentionNorm);
        if (model.getConfig().residualMultiplier != null) {
            configurableTensorProvider.get().scale(model.getConfig().residualMultiplier, lnattn, 0, model.getConfig().embeddingLength);
        }
        configurableTensorProvider.get().accumulate(lnattn, embedding, 0, model.getConfig().embeddingLength);
        logIfInteresting("post_attn_residual", lnattn);

        AbstractTensor lnpreFF = preFFNorm.map(ln -> ln.forward(lnattn)).orElse(lnattn);
        AbstractTensor postFF;
        try (AbstractTensor qlnemb2 = model.maybeQuantize(lnpreFF)) {
            postFF = ffBlock.forward(qlnemb2, tensorReducer);
        }

        AbstractTensor lnpostFF;
        if (hasMoeBlock()) {
            lnpostFF = forwardMoeFeedForward(postFF, lnattn);
        } else {
            lnpostFF = maybeApplyNorm(postFF, postFFNorm);
        }
        if (model.getConfig().residualMultiplier != null) {
            configurableTensorProvider.get().scale(model.getConfig().residualMultiplier, lnpostFF, 0, model.getConfig().embeddingLength);
        }
        configurableTensorProvider.get().accumulate(lnpostFF, lnattn, 0, model.getConfig().embeddingLength);
        logIfInteresting("post_ff_residual", lnpostFF);

        if (perLayerInput != null && perLayerInputGateWeights.isPresent() && perLayerProjectionWeights.isPresent() && postPerLayerInputNorm.isPresent()) {
            try (
                    AbstractTensor gated = model.makeDenseTensor(lnpostFF.shape().first(), perLayerInputLength);
                    AbstractTensor projected = model.makeDenseTensor(lnpostFF.shape().first(), model.getConfig().embeddingLength)
            ) {
                configurableTensorProvider.get().dotProductChunk(gated, lnpostFF, perLayerInputGateWeights.get(), 0,
                        model.getConfig().embeddingLength, 0, perLayerInputLength);
                for (int b = 0; b < gated.shape().first(); b++) {
                    for (int i = 0; i < perLayerInputLength; i++) {
                        float v = ActivationFunction.eval(activationFunction, gated.get(b, i)) * perLayerInput.get(b, i);
                        gated.set(v, b, i);
                    }
                }
                try (AbstractTensor gatedQ = model.maybeQuantize(gated)) {
                    configurableTensorProvider.get().dotProductChunk(projected, gatedQ, perLayerProjectionWeights.get(), 0,
                            perLayerInputLength, 0, model.getConfig().embeddingLength);
                }
                AbstractTensor normalized = postPerLayerInputNorm.get().forward(projected);
                configurableTensorProvider.get().accumulate(normalized, lnpostFF, 0, model.getConfig().embeddingLength);
                lnpostFF.close();
                lnpostFF = normalized;
            }
            logIfInteresting("post_ple", lnpostFF);
        }

        if (layerScalar != 1.0f) {
            configurableTensorProvider.get().scale(layerScalar, lnpostFF, 0, model.getConfig().embeddingLength);
        }
        logIfInteresting("final", lnpostFF);

        if (lnemb != embedding) lnemb.close();
        if (lnattn != postAttention) lnattn.close();
        else postAttention.close();
        if (lnpreFF != lnattn) lnpreFF.close();
        else lnattn.close();

        return lnpostFF;
        } finally {
            totalTimer.stop();
        }
    }

    private AbstractTensor maybeApplyNorm(AbstractTensor tensor, Optional<LayerNorm> norm) {
        return norm.map(ln -> {
            AbstractTensor o = ln.forward(tensor);
            tensor.close();
            return o;
        }).orElse(tensor);
    }

    private boolean hasMoeBlock() {
        return postFeedForwardNorm1.isPresent()
                && preFeedForwardNorm2.isPresent()
                && postFeedForwardNorm2.isPresent()
                && routerProjectionWeights.isPresent()
                && routerScaleWeights.isPresent()
                && routerPerExpertScaleWeights.isPresent()
                && expertGateUpWeights.isPresent()
                && expertDownWeights.isPresent()
                && numberOfExperts > 0
                && topKExperts > 0
                && moeIntermediateLength > 0;
    }

    private AbstractTensor forwardMoeFeedForward(AbstractTensor denseMlpOutput, AbstractTensor feedForwardResidual) {
        AbstractTensor denseBranch = postFeedForwardNorm1.get().forward(denseMlpOutput);
        denseMlpOutput.close();
        try (AbstractTensor expertInput = preFeedForwardNorm2.get().forward(feedForwardResidual);
             AbstractTensor expertOutput = forwardExperts(feedForwardResidual, expertInput)) {
            AbstractTensor expertBranch = postFeedForwardNorm2.get().forward(expertOutput);
            configurableTensorProvider.get().accumulate(denseBranch, expertBranch, 0, model.getConfig().embeddingLength);
            expertBranch.close();
        }
        return maybeApplyNorm(denseBranch, postFFNorm);
    }

    private AbstractTensor forwardExperts(AbstractTensor routerInput, AbstractTensor expertInput) {
        int batchSize = routerInput.shape().first();
        int hiddenSize = model.getConfig().embeddingLength;
        AbstractTensor output = model.makeDenseTensor(batchSize, hiddenSize);
        try (AbstractTensor normalizedRouterInput = model.makeDenseTensor(batchSize, hiddenSize);
             AbstractTensor expertHidden = model.makeDenseTensor(1, moeIntermediateLength)) {
            normalizeRouterInput(routerInput, normalizedRouterInput, hiddenSize);
            for (int b = 0; b < batchSize; b++) {
                float[] probabilities = routerProbabilities(normalizedRouterInput, b, hiddenSize);
                int[] selectedExperts = topK(probabilities);
                float selectedSum = 0.0f;
                for (int expert : selectedExperts) {
                    selectedSum += probabilities[expert];
                }
                for (int expert : selectedExperts) {
                    float weight = probabilities[expert] / selectedSum;
                    weight *= routerPerExpertScaleWeights.get().get(0, expert);
                    computeExpertHidden(expertInput, b, expert, hiddenSize, expertHidden);
                    accumulateExpertDownProjection(output, b, expert, weight, hiddenSize, expertHidden);
                }
            }
        }
        return output;
    }

    private void normalizeRouterInput(AbstractTensor input, AbstractTensor output, int hiddenSize) {
        float rootScale = (float) FastMath.pow(hiddenSize, -0.5);
        for (int b = 0; b < input.shape().first(); b++) {
            double sumSquares = 0.0d;
            for (int h = 0; h < hiddenSize; h++) {
                float v = input.get(b, h);
                sumSquares += (double) v * v;
            }
            float invRms = (float) (1.0d / FastMath.sqrt(sumSquares / hiddenSize + model.getConfig().layerNormEps));
            for (int h = 0; h < hiddenSize; h++) {
                output.set(input.get(b, h) * invRms * routerScaleWeights.get().get(0, h) * rootScale, b, h);
            }
        }
    }

    private float[] routerProbabilities(AbstractTensor normalizedRouterInput, int batchIndex, int hiddenSize) {
        float[] scores = new float[numberOfExperts];
        float max = Float.NEGATIVE_INFINITY;
        for (int expert = 0; expert < numberOfExperts; expert++) {
            float score = 0.0f;
            for (int h = 0; h < hiddenSize; h++) {
                score += normalizedRouterInput.get(batchIndex, h) * routerProjectionWeights.get().get(expert, h);
            }
            scores[expert] = score;
            max = Math.max(max, score);
        }
        double sum = 0.0d;
        for (int expert = 0; expert < numberOfExperts; expert++) {
            scores[expert] = (float) FastMath.exp(scores[expert] - max);
            sum += scores[expert];
        }
        for (int expert = 0; expert < numberOfExperts; expert++) {
            scores[expert] /= (float) sum;
        }
        return scores;
    }

    private int[] topK(float[] probabilities) {
        int[] selected = new int[topKExperts];
        for (int i = 0; i < selected.length; i++) {
            selected[i] = i;
        }
        for (int expert = topKExperts; expert < probabilities.length; expert++) {
            int minIndex = 0;
            for (int i = 1; i < selected.length; i++) {
                if (probabilities[selected[i]] < probabilities[selected[minIndex]]) {
                    minIndex = i;
                }
            }
            if (probabilities[expert] > probabilities[selected[minIndex]]) {
                selected[minIndex] = expert;
            }
        }
        return selected;
    }

    private void computeExpertHidden(AbstractTensor expertInput, int batchIndex, int expert, int hiddenSize,
            AbstractTensor expertHidden) {
        for (int intermediate = 0; intermediate < moeIntermediateLength; intermediate++) {
            float gate = 0.0f;
            float up = 0.0f;
            for (int h = 0; h < hiddenSize; h++) {
                float input = expertInput.get(batchIndex, h);
                gate += input * expertGateUpWeight(expert, intermediate, h, hiddenSize);
                up += input * expertGateUpWeight(expert, moeIntermediateLength + intermediate, h, hiddenSize);
            }
            expertHidden.set(ActivationFunction.eval(activationFunction, gate) * up, 0, intermediate);
        }
    }

    private void accumulateExpertDownProjection(AbstractTensor output, int batchIndex, int expert, float expertWeight,
            int hiddenSize, AbstractTensor expertHidden) {
        for (int h = 0; h < hiddenSize; h++) {
            float value = 0.0f;
            for (int intermediate = 0; intermediate < moeIntermediateLength; intermediate++) {
                value += expertHidden.get(0, intermediate) * expertDownWeight(expert, h, intermediate, hiddenSize);
            }
            output.set(output.get(batchIndex, h) + value * expertWeight, batchIndex, h);
        }
    }

    private float expertGateUpWeight(int expert, int row, int column, int hiddenSize) {
        AbstractTensor weights = expertGateUpWeights.get();
        if (weights.shape().dims() == 3) {
            return weights.get(expert, row, column);
        }
        return weights.get(expert, row * hiddenSize + column);
    }

    private float expertDownWeight(int expert, int row, int column, int hiddenSize) {
        AbstractTensor weights = expertDownWeights.get();
        if (weights.shape().dims() == 3) {
            return weights.get(expert, row, column);
        }
        return weights.get(expert, row * moeIntermediateLength + column);
    }



    private void logIfInteresting(String stage, AbstractTensor tensor) {
        if (!DEBUG_BLOCK_SUMMARIES || (layerIndex != 4 && layerIndex != 34)) {
            return;
        }
        int width = tensor.shape().last();
        float min = Float.POSITIVE_INFINITY;
        float max = Float.NEGATIVE_INFINITY;
        double sum = 0.0d;
        double sumSquares = 0.0d;
        StringBuilder first = new StringBuilder();
        int preview = Math.min(8, width);
        for (int i = 0; i < width; i++) {
            float v = tensor.get(0, i);
            if (v < min) min = v;
            if (v > max) max = v;
            sum += v;
            sumSquares += (double) v * v;
            if (i < preview) {
                if (i > 0) first.append(',');
                first.append(String.format(Locale.ROOT, "%.4f", v));
            }
        }
        double mean = sum / width;
        double l2 = Math.sqrt(sumSquares);
        logger.info("gemma4 block_summary layer={} stage={} row0_min={} row0_max={} row0_mean={} row0_l2={} row0_first8=[{}]",
                layerIndex,
                stage,
                String.format(Locale.ROOT, "%.6f", min),
                String.format(Locale.ROOT, "%.6f", max),
                String.format(Locale.ROOT, "%.6f", mean),
                String.format(Locale.ROOT, "%.6f", l2),
                first);
    }
}
