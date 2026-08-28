package io.teknek.deliverance.model.diffusiongemma;

import com.google.common.base.Preconditions;
import io.dropwizard.metrics5.MetricRegistry;
import io.dropwizard.metrics5.Timer;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorMutability;
import io.teknek.deliverance.tensor.TensorProbability;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.operations.TensorOperations;

import java.util.Objects;

/**
 * Stops DiffusionGemma denoising when the canvas is both stable and confident.
 *
 * <p>The confidence condition is based on model uncertainty: token entropy is computed for every canvas position and the
 * mean entropy for each batch row must be below {@code confidenceThreshold}. The stability condition is based on canvas
 * movement: the argmax canvas must remain unchanged across enough consecutive denoising steps. The final decision is the
 * conjunction of both conditions for each batch row.</p>
 */
public final class StableAndConfidentStoppingCriteria {
    private final int stabilityThreshold;
    private final float confidenceThreshold;
    private final TensorOperations tensorOperations;
    private final MetricRegistry metricRegistry;
    private AbstractTensor previousArgmaxCanvas;
    private int[] stableCounts;

    public StableAndConfidentStoppingCriteria(int stabilityThreshold, float confidenceThreshold,
            TensorOperations tensorOperations, MetricRegistry metricRegistry) {
        Preconditions.checkArgument(stabilityThreshold >= 0, "stabilityThreshold must be >= 0");
        Preconditions.checkArgument(Float.isFinite(confidenceThreshold) && confidenceThreshold > 0.0f,
                "confidenceThreshold must be finite and > 0");
        this.stabilityThreshold = stabilityThreshold;
        this.confidenceThreshold = confidenceThreshold;
        this.tensorOperations = Objects.requireNonNull(tensorOperations, "tensorOperations");
        this.metricRegistry = Objects.requireNonNull(metricRegistry, "metricRegistry");
    }

    /**
     * Writes one stop decision per batch row into {@code output}.
     *
     * <p>{@code output[batch, 0]} is {@code 1.0} when that batch row should stop denoising and {@code 0.0} otherwise.
     * {@code argmaxCanvas} is the current model-selected token canvas, normally shaped {@code [batch, canvasLength]}.
     * {@code logits} must be shaped {@code [batch, canvasLength, vocabSize]}.</p>
     */
    public void shouldStop(AbstractTensor output, AbstractTensor argmaxCanvas, AbstractTensor logits) {
        try (Timer.Context ignored = InferenceProfiler.timer(metricRegistry,
                "diffusiongemma.stopping.should_stop").time()) {
            validateOutput(output, argmaxCanvas);
            validateArgmaxCanvas(argmaxCanvas);
            validateLogits(logits, argmaxCanvas);
            TensorMutability.requireWritable(output, "shouldStop");
            int batchSize = (int) argmaxCanvas.shape().first();
            int canvasLength = (int) argmaxCanvas.shape().last();
            ensureState(argmaxCanvas);

            try (FloatBufferTensor tokenEntropy = new FloatBufferTensor(batchSize, canvasLength)) {
                try (Timer.Context ignoredEntropy = InferenceProfiler.timer(metricRegistry,
                        "diffusiongemma.stopping.entropy").time()) {
                    TensorProbability.entropy(tokenEntropy, logits, tensorOperations);
                }
                try (Timer.Context ignoredDecision = InferenceProfiler.timer(metricRegistry,
                        "diffusiongemma.stopping.decision").time()) {
                    int stopCount = 0;
                    for (int batch = 0; batch < batchSize; batch++) {
                        boolean stable = isStable(batch, argmaxCanvas);
                        stableCounts[batch] = stable ? stableCounts[batch] + 1 : 0;
                        boolean confident = meanEntropy(tokenEntropy, batch, canvasLength) < confidenceThreshold;
                        boolean stop = stableCounts[batch] >= stabilityThreshold && confident;
                        output.set(stop ? 1.0f : 0.0f, batch, 0);
                        if (stop) {
                            stopCount++;
                        }
                    }
                    previousArgmaxCanvas.copyFrom(argmaxCanvas, 0, 0, (int) argmaxCanvas.size());
                    if (InferenceProfiler.isEnabled()) {
                        InferenceProfiler.counter(metricRegistry, "diffusiongemma.stopping.stop_true").inc(stopCount);
                        InferenceProfiler.counter(metricRegistry, "diffusiongemma.stopping.stop_false")
                                .inc(batchSize - stopCount);
                    }
                }
            }
        }
    }

    private void ensureState(AbstractTensor argmaxCanvas) {
        int batchSize = (int) argmaxCanvas.shape().first();
        if (previousArgmaxCanvas != null && previousArgmaxCanvas.shape().equals(argmaxCanvas.shape())) {
            return;
        }
        if (previousArgmaxCanvas != null) {
            previousArgmaxCanvas.close();
        }
        previousArgmaxCanvas = new FloatBufferTensor(argmaxCanvas.shape());
        stableCounts = new int[batchSize];
    }

    private boolean isStable(int batch, AbstractTensor argmaxCanvas) {
        for (int position = 0; position < argmaxCanvas.shape().last(); position++) {
            if (argmaxCanvas.get(batch, position) != previousArgmaxCanvas.get(batch, position)) {
                return false;
            }
        }
        return true;
    }

    private static float meanEntropy(AbstractTensor tokenEntropy, int batch, int canvasLength) {
        float sum = 0.0f;
        for (int position = 0; position < canvasLength; position++) {
            sum += tokenEntropy.get(batch, position);
        }
        return sum / canvasLength;
    }

    private static void validateOutput(AbstractTensor output, AbstractTensor argmaxCanvas) {
        Preconditions.checkArgument(output.dims() == 2 && output.shape().last() == 1,
                "output must have shape [batch, 1]");
        Preconditions.checkArgument(output.shape().first() == argmaxCanvas.shape().first(),
                "output batch must match argmaxCanvas batch");
        Preconditions.checkArgument(output.dType() == DType.F32, "output must be F32");
    }

    private static void validateArgmaxCanvas(AbstractTensor argmaxCanvas) {
        Preconditions.checkArgument(argmaxCanvas.dims() == 2 && argmaxCanvas.shape().first() > 0
                        && argmaxCanvas.shape().last() > 0,
                "argmaxCanvas must have shape [batch, canvasLength]");
        Preconditions.checkArgument(argmaxCanvas.dType() == DType.F32, "argmaxCanvas must be F32 token-id tensor");
    }

    private static void validateLogits(AbstractTensor logits, AbstractTensor argmaxCanvas) {
        Preconditions.checkArgument(logits.dims() == 3, "logits must have shape [batch, canvasLength, vocab]");
        Preconditions.checkArgument(logits.shape().dim(0) == argmaxCanvas.shape().first()
                        && logits.shape().dim(1) == argmaxCanvas.shape().last(),
                "logits batch/canvas dimensions must match argmaxCanvas");
    }
}
