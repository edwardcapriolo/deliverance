package io.teknek.deliverance.model.diffusiongemma;

import com.google.common.base.Preconditions;
import io.dropwizard.metrics5.MetricRegistry;
import io.dropwizard.metrics5.Timer;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorMutability;
import io.teknek.deliverance.tensor.operations.TensorOperations;

import java.util.Objects;

/**
 * Applies DiffusionGemma's linear denoising temperature schedule to logits in place.
 *
 * <p>DiffusionGemma denoises a whole canvas over multiple refinement steps instead of sampling one autoregressive token
 * at a time. Early denoising steps use a higher temperature so the canvas can still move, while later steps use a lower
 * temperature so token distributions sharpen before entropy-based acceptance and stopping checks. Hugging Face names the
 * endpoints {@code t_max} for the first/noisiest step and {@code t_min} for the final/most confident step, but the public
 * processor is called with a decreasing {@code curStep}: {@code maxDenoisingSteps} maps to {@code tMax}, halfway maps to
 * the midpoint, and {@code 0} maps to {@code tMin}.</p>
 *
 * <p>The processor divides logits by the scheduled temperature by scaling them with {@code 1 / temperature}. This matches
 * standard temperature scaling: temperatures below {@code 1.0} sharpen the distribution before softmax/entropy, and
 * temperatures above {@code 1.0} flatten it.</p>
 */
public final class LinearTemperatureScheduleLogitsProcessor implements DiffusionGemmaLogitsProcessor {
    private final float tMin;
    private final float tMax;
    private final int maxDenoisingSteps;
    private final TensorOperations tensorOperations;
    private final MetricRegistry metricRegistry;

    public LinearTemperatureScheduleLogitsProcessor(float tMin, float tMax, int maxDenoisingSteps,
            TensorOperations tensorOperations, MetricRegistry metricRegistry) {
        Preconditions.checkArgument(Float.isFinite(tMin) && tMin >= 0.0f, "tMin must be finite and >= 0");
        Preconditions.checkArgument(Float.isFinite(tMax) && tMax > tMin, "tMax must be finite and > tMin");
        Preconditions.checkArgument(maxDenoisingSteps > 0, "maxDenoisingSteps must be > 0");
        this.tMin = tMin;
        this.tMax = tMax;
        this.maxDenoisingSteps = maxDenoisingSteps;
        this.tensorOperations = Objects.requireNonNull(tensorOperations, "tensorOperations");
        this.metricRegistry = Objects.requireNonNull(metricRegistry, "metricRegistry");
    }

    @Override
    public void process(AbstractTensor logits, int curStep) {
        try (Timer.Context ignored = InferenceProfiler.timer(metricRegistry,
                "diffusiongemma.logits_processor.linear_temperature").time()) {
            TensorMutability.requireWritable(logits, "linearTemperatureSchedule");
            Preconditions.checkArgument(curStep >= 0 && curStep <= maxDenoisingSteps,
                    "curStep must be in [0, maxDenoisingSteps]");
            float temperature = temperature(curStep);
            tensorOperations.scale(1.0f / temperature, logits, 0, (int) logits.shape().last());
        }
    }

    /**
     * Returns the linear temperature for the current denoising step.
     *
     * <p>The formula intentionally follows the Hugging Face implementation:</p>
     *
     * <pre>{@code
     * temperature = tMin + (tMax - tMin) * curStep / maxDenoisingSteps
     * }</pre>
     *
     * <p>Therefore {@code curStep == maxDenoisingSteps} applies {@code tMax}, {@code curStep == 0} applies {@code tMin},
     * and intermediate steps linearly interpolate between them.</p>
     */
    float temperature(int curStep) {
        return tMin + (tMax - tMin) * curStep / maxDenoisingSteps;
    }
}
