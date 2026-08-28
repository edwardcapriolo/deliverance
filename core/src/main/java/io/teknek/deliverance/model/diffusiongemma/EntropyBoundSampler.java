package io.teknek.deliverance.model.diffusiongemma;

import io.dropwizard.metrics5.MetricRegistry;
import io.dropwizard.metrics5.Timer;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorProbability;
import io.teknek.deliverance.tensor.TensorMutability;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.operations.TensorOperations;

import java.util.Arrays;
import java.util.Objects;
import java.util.Random;

public final class EntropyBoundSampler {
    private final float entropyBound;
    private final int canvasLength;
    private final int vocabSize;
    private final Random random;
    private final TensorOperations tensorOperations;
    private final MetricRegistry metricRegistry;
    private boolean[][] acceptedTokenMask;

    public EntropyBoundSampler(float entropyBound, int canvasLength, int vocabSize, Random random,
            TensorOperations tensorOperations, MetricRegistry metricRegistry) {
        if (!Float.isFinite(entropyBound) || entropyBound <= 0.0f) {
            throw new IllegalArgumentException("entropyBound must be finite and > 0");
        }
        if (canvasLength <= 0) {
            throw new IllegalArgumentException("canvasLength must be > 0");
        }
        if (vocabSize <= 0) {
            throw new IllegalArgumentException("vocabSize must be > 0");
        }
        this.entropyBound = entropyBound;
        this.canvasLength = canvasLength;
        this.vocabSize = vocabSize;
        this.random = Objects.requireNonNull(random, "random");
        this.tensorOperations = Objects.requireNonNull(tensorOperations, "tensorOperations");
        this.metricRegistry = Objects.requireNonNull(metricRegistry, "metricRegistry");
    }

    public void initializeCanvas(AbstractTensor canvas) {
        validateCanvas(canvas, "canvas");
        TensorMutability.requireWritable(canvas, "initializeCanvas");
        for (int batch = 0; batch < canvas.shape().first(); batch++) {
            for (int position = 0; position < canvasLength; position++) {
                canvas.set(random.nextInt(vocabSize), batch, position);
            }
        }
    }

    /**
     * Accepts proposed denoiser tokens for the lowest-entropy canvas positions.
     *
     * <p>The current canvas contains the tokens from the previous denoising step. The denoiser canvas contains the
     * model's proposed replacement tokens for this step. This method computes one entropy value per canvas position from
     * logits shaped {@code [batch, canvasLength, vocabSize]}, sorts positions from lowest entropy to highest entropy, and
     * accepts positions while {@code cumulativeEntropy - currentEntropy <= entropyBound}. Accepted positions take their
     * token from {@code denoiserCanvas}; rejected positions keep their token from {@code currentCanvas}.</p>
     *
     * <p>The accepted-position mask is stored on the sampler so {@link #renoiseCanvas(AbstractTensor, AbstractTensor, int)}
     * can keep accepted tokens fixed and re-randomize the rejected positions.</p>
     */
    public void acceptCanvas(AbstractTensor acceptedCanvas, AbstractTensor currentCanvas, AbstractTensor denoiserCanvas,
            AbstractTensor logits, int curStep) {
        try (Timer.Context ignored = InferenceProfiler.timer(metricRegistry,
                "diffusiongemma.sampler.accept_canvas").time()) {
        validateCanvas(acceptedCanvas, "acceptedCanvas");
        validateCanvas(currentCanvas, "currentCanvas");
        validateCanvas(denoiserCanvas, "denoiserCanvas");
        TensorMutability.requireWritable(acceptedCanvas, "acceptCanvas");
        if (logits.dims() != 3 || logits.shape().dim(0) != currentCanvas.shape().first()
                || logits.shape().dim(1) != canvasLength || logits.shape().dim(2) != vocabSize) {
            throw new IllegalArgumentException("logits must have shape [batchSize, canvasLength, vocabSize]");
        }
        int batchSize = (int) currentCanvas.shape().first();
        boolean[][] accepted = new boolean[batchSize][canvasLength];
        acceptedCanvas.copyFrom(currentCanvas, 0, 0, (int) currentCanvas.size());
        int acceptedCount = 0;
        try (FloatBufferTensor tokenEntropy = new FloatBufferTensor(batchSize, canvasLength)) {
            try (Timer.Context ignoredEntropy = InferenceProfiler.timer(metricRegistry,
                    "diffusiongemma.sampler.entropy").time()) {
                TensorProbability.entropy(tokenEntropy, logits, tensorOperations);
            }
            try (Timer.Context ignoredSelection = InferenceProfiler.timer(metricRegistry,
                    "diffusiongemma.sampler.accept_selection").time()) {
            for (int batch = 0; batch < batchSize; batch++) {
                PositionEntropy[] entropies = new PositionEntropy[canvasLength];
                for (int position = 0; position < canvasLength; position++) {
                    entropies[position] = new PositionEntropy(position, tokenEntropy.get(batch, position));
                }
                Arrays.sort(entropies);
                double cumulativeEntropy = 0.0;
                for (PositionEntropy positionEntropy : entropies) {
                    cumulativeEntropy += positionEntropy.entropy;
                    if (cumulativeEntropy - positionEntropy.entropy <= entropyBound) {
                        int position = positionEntropy.position;
                        accepted[batch][position] = true;
                        acceptedCount++;
                        acceptedCanvas.set(denoiserCanvas.get(batch, position), batch, position);
                    }
                }
            }
            }
        }
        if (InferenceProfiler.isEnabled()) {
            InferenceProfiler.counter(metricRegistry, "diffusiongemma.sampler.accepted_tokens").inc(acceptedCount);
            InferenceProfiler.counter(metricRegistry, "diffusiongemma.sampler.rejected_tokens")
                    .inc((long) batchSize * canvasLength - acceptedCount);
        }
        this.acceptedTokenMask = accepted;
        }
    }

    public void renoiseCanvas(AbstractTensor renoisedCanvas, AbstractTensor acceptedCanvas, int curStep) {
        try (Timer.Context ignored = InferenceProfiler.timer(metricRegistry,
                "diffusiongemma.sampler.renoise_canvas").time()) {
        if (acceptedTokenMask == null) {
            throw new IllegalStateException("acceptCanvas must be called before renoiseCanvas");
        }
        validateCanvas(renoisedCanvas, "renoisedCanvas");
        validateCanvas(acceptedCanvas, "acceptedCanvas");
        TensorMutability.requireWritable(renoisedCanvas, "renoiseCanvas");
        int batchSize = (int) acceptedCanvas.shape().first();
        if (acceptedTokenMask.length != batchSize) {
            throw new IllegalStateException("acceptedTokenMask batch size does not match acceptedCanvas");
        }
        for (int batch = 0; batch < batchSize; batch++) {
            if (acceptedTokenMask[batch].length != canvasLength) {
                throw new IllegalStateException("acceptedTokenMask row " + batch + " does not match canvasLength");
            }
            for (int position = 0; position < canvasLength; position++) {
                if (acceptedTokenMask[batch][position]) {
                    renoisedCanvas.set(acceptedCanvas.get(batch, position), batch, position);
                } else {
                    renoisedCanvas.set(random.nextInt(vocabSize), batch, position);
                }
            }
        }
        }
    }

    private void validateCanvas(AbstractTensor canvas, String name) {
        Objects.requireNonNull(canvas, name);
        if (canvas.dims() != 2 || canvas.shape().first() <= 0 || canvas.shape().last() != canvasLength) {
            throw new IllegalArgumentException(name + " must have shape [batchSize, " + canvasLength + "]");
        }
        if (canvas.dType() != DType.F32) {
            throw new IllegalArgumentException(name + " must be F32 token-id tensor");
        }
    }

    private record PositionEntropy(int position, double entropy) implements Comparable<PositionEntropy> {
        @Override
        public int compareTo(PositionEntropy other) {
            int entropyComparison = Double.compare(this.entropy, other.entropy);
            return entropyComparison != 0 ? entropyComparison : Integer.compare(this.position, other.position);
        }
    }
}
