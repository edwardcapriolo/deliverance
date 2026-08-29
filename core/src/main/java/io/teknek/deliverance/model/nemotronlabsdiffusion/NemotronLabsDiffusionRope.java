package io.teknek.deliverance.model.nemotronlabsdiffusion;

import com.google.common.base.Preconditions;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.operations.TensorOperations;
import net.jafama.FastMath;

import java.util.Map;

/**
 * Nemotron/Ministral RoPE helper matching upstream Transformers YaRN formulas.
 *
 * <p>Formula source: `transformers.modeling_rope_utils._compute_yarn_parameters` and
 * `modeling_ministral.apply_rotary_pos_emb` from `nvidia/Nemotron-Labs-Diffusion-3B-Base`.</p>
 */
final class NemotronLabsDiffusionRope {
    private final int headDim;
    private final int rotaryDim;
    private final float[] invFreq;
    private final float attentionScaling;
    private final float llama4ScalingBeta;
    private final int originalMaxPositionEmbeddings;

    NemotronLabsDiffusionRope(NemotronLabsDiffusionConfig config) {
        this(config.headSize, config.embeddingLength, config.numberOfHeads, config.contextLength,
                config.ropeParameters);
    }

    NemotronLabsDiffusionRope(int headDim, int hiddenSize, int numberOfHeads, int maxPositionEmbeddings,
            Map<String, Object> ropeParameters) {
        this.headDim = headDim;
        this.rotaryDim = (int) (headDim * number(ropeParameters, "partial_rotary_factor", 1.0d));
        Preconditions.checkArgument(rotaryDim > 0 && rotaryDim % 2 == 0, "rotaryDim must be positive and even");
        YarnParameters yarn = computeYarnParameters(headDim, hiddenSize, numberOfHeads, maxPositionEmbeddings,
                ropeParameters);
        this.invFreq = yarn.invFreq();
        this.attentionScaling = yarn.attentionScaling();
        this.llama4ScalingBeta = (float) number(ropeParameters, "llama_4_scaling_beta", 0.0d);
        this.originalMaxPositionEmbeddings = (int) number(ropeParameters, "original_max_position_embeddings",
                maxPositionEmbeddings);
    }

    int rotaryDim() {
        return rotaryDim;
    }

    float attentionScaling() {
        return attentionScaling;
    }

    float invFreq(int index) {
        return invFreq[index];
    }

    int invFreqLength() {
        return invFreq.length;
    }

    float llama4QueryScale(int position) {
        if (llama4ScalingBeta == 0.0f) {
            return 1.0f;
        }
        return (float) (1.0 + llama4ScalingBeta
                * StrictMath.log(1.0 + StrictMath.floor((double) position / originalMaxPositionEmbeddings)));
    }

    void apply(AbstractTensor query, AbstractTensor key, int queryHeads, int keyValueHeads, TensorOperations ops) {
        apply(query, key, 0, queryHeads, keyValueHeads, ops);
    }

    void apply(AbstractTensor query, AbstractTensor key, int startPosition, int queryHeads, int keyValueHeads,
            TensorOperations ops) {
        Preconditions.checkArgument(query.dims() == 2 && key.dims() == 2, "query/key must be 2D");
        Preconditions.checkArgument(query.shape().first() == key.shape().first(), "query/key rows must match");
        for (int row = 0; row < query.shape().first(); row++) {
            int absolutePosition = startPosition + row;
            applyToTensor(query, row, absolutePosition, queryHeads);
            applyToTensor(key, row, absolutePosition, keyValueHeads);
            float queryScale = llama4QueryScale(absolutePosition);
            if (queryScale != 1.0f) {
                try (AbstractTensor queryRow = query.slice(row)) {
                    ops.scale(queryScale, queryRow, 0, queryHeads * headDim);
                }
            }
        }
    }

    private void applyToTensor(AbstractTensor tensor, int row, int absolutePosition, int heads) {
        int half = rotaryDim / 2;
        for (int i = 0; i < half; i++) {
            float angle = absolutePosition * invFreq[i];
            float cos = (float) FastMath.cos(angle) * attentionScaling;
            float sin = (float) FastMath.sin(angle) * attentionScaling;
            for (int head = 0; head < heads; head++) {
                int headOffset = head * headDim;
                int xOffset = headOffset + i;
                int yOffset = headOffset + i + half;
                float x = tensor.get(row, xOffset);
                float y = tensor.get(row, yOffset);
                tensor.set(x * cos - y * sin, row, xOffset);
                tensor.set(y * cos + x * sin, row, yOffset);
            }
        }
    }

    static YarnParameters computeYarnParameters(int headDim, int hiddenSize, int numberOfHeads,
            int maxPositionEmbeddings, Map<String, Object> ropeParameters) {
        double base = number(ropeParameters, "rope_theta", 10_000.0d);
        double partialRotaryFactor = number(ropeParameters, "partial_rotary_factor", 1.0d);
        int effectiveHeadDim = headDim > 0 ? headDim : hiddenSize / numberOfHeads;
        int dim = (int) (effectiveHeadDim * partialRotaryFactor);
        double factor = number(ropeParameters, "factor", Double.NaN);
        int originalMaxPositionEmbeddings = (int) number(ropeParameters, "original_max_position_embeddings",
                maxPositionEmbeddings);
        if (Double.isNaN(factor)) {
            factor = (double) maxPositionEmbeddings / originalMaxPositionEmbeddings;
        }

        double attentionFactor = attentionFactor(ropeParameters, factor);
        double betaFast = number(ropeParameters, "beta_fast", 32.0d);
        double betaSlow = number(ropeParameters, "beta_slow", 1.0d);
        boolean truncate = booleanValue(ropeParameters, "truncate", true);
        double low = findCorrectionDim(betaFast, dim, base, originalMaxPositionEmbeddings);
        double high = findCorrectionDim(betaSlow, dim, base, originalMaxPositionEmbeddings);
        if (truncate) {
            low = StrictMath.floor(low);
            high = StrictMath.ceil(high);
        }
        low = Math.max(low, 0.0d);
        high = Math.min(high, dim - 1.0d);

        int length = dim / 2;
        float[] invFreq = new float[length];
        for (int i = 0; i < length; i++) {
            double exponent = (2.0d * i) / dim;
            double posFreq = FastMath.pow(base, exponent);
            double extrapolation = 1.0d / posFreq;
            double interpolation = 1.0d / (factor * posFreq);
            double extrapolationFactor = 1.0d - linearRampFactor(i, low, high);
            invFreq[i] = (float) (interpolation * (1.0d - extrapolationFactor)
                    + extrapolation * extrapolationFactor);
        }
        return new YarnParameters(invFreq, (float) attentionFactor);
    }

    private static double attentionFactor(Map<String, Object> ropeParameters, double factor) {
        Object explicit = ropeParameters.get("attention_factor");
        if (explicit instanceof Number number) {
            return number.doubleValue();
        }
        Object mscale = ropeParameters.get("mscale");
        Object mscaleAllDim = ropeParameters.get("mscale_all_dim");
        if (mscale instanceof Number mscaleNumber && mscaleAllDim instanceof Number mscaleAllDimNumber) {
            return getMscale(factor, mscaleNumber.doubleValue()) / getMscale(factor, mscaleAllDimNumber.doubleValue());
        }
        return getMscale(factor, 1.0d);
    }

    private static double getMscale(double scale, double mscale) {
        if (scale <= 1.0d) {
            return 1.0d;
        }
        return 0.1d * mscale * StrictMath.log(scale) + 1.0d;
    }

    private static double findCorrectionDim(double numRotations, int dim, double base, int maxPositionEmbeddings) {
        return (dim * StrictMath.log(maxPositionEmbeddings / (numRotations * 2.0d * StrictMath.PI)))
                / (2.0d * StrictMath.log(base));
    }

    private static double linearRampFactor(double value, double min, double max) {
        if (min == max) {
            max += 0.001d;
        }
        return Math.max(0.0d, Math.min(1.0d, (value - min) / (max - min)));
    }

    private static double number(Map<String, Object> values, String key, double fallback) {
        Object value = values.get(key);
        return value instanceof Number number ? number.doubleValue() : fallback;
    }

    private static boolean booleanValue(Map<String, Object> values, String key, boolean fallback) {
        Object value = values.get(key);
        return value instanceof Boolean bool ? bool : fallback;
    }

    record YarnParameters(float[] invFreq, float attentionScaling) {
    }
}
