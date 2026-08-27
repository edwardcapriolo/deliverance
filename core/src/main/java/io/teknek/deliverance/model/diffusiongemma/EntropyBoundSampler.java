package io.teknek.deliverance.model.diffusiongemma;

import io.teknek.deliverance.tensor.AbstractTensor;
import net.jafama.FastMath;

import java.util.Arrays;
import java.util.Objects;
import java.util.Random;

public final class EntropyBoundSampler {
    private final float entropyBound;
    private final int canvasLength;
    private final int vocabSize;
    private final Random random;
    private boolean[][] acceptedTokenMask;

    public EntropyBoundSampler(float entropyBound, int canvasLength, int vocabSize, Random random) {
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
    }

    public int[][] initializeCanvas(int batchSize) {
        if (batchSize <= 0) {
            throw new IllegalArgumentException("batchSize must be > 0");
        }
        int[][] canvas = new int[batchSize][canvasLength];
        for (int batch = 0; batch < batchSize; batch++) {
            for (int position = 0; position < canvasLength; position++) {
                canvas[batch][position] = random.nextInt(vocabSize);
            }
        }
        return canvas;
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
     * <p>The accepted-position mask is stored on the sampler so {@link #renoiseCanvas(int[][], int)} can keep accepted
     * tokens fixed and re-randomize the rejected positions.</p>
     */
    public int[][] acceptCanvas(int[][] currentCanvas, int[][] denoiserCanvas, AbstractTensor logits, int curStep) {
        validateCanvas(currentCanvas, "currentCanvas");
        validateCanvas(denoiserCanvas, "denoiserCanvas");
        if (logits.dims() != 3 || logits.shape().dim(0) != currentCanvas.length
                || logits.shape().dim(1) != canvasLength || logits.shape().dim(2) != vocabSize) {
            throw new IllegalArgumentException("logits must have shape [batchSize, canvasLength, vocabSize]");
        }
        int batchSize = currentCanvas.length;
        boolean[][] accepted = new boolean[batchSize][canvasLength];
        int[][] result = copyCanvas(currentCanvas);
        for (int batch = 0; batch < batchSize; batch++) {
            PositionEntropy[] entropies = new PositionEntropy[canvasLength];
            for (int position = 0; position < canvasLength; position++) {
                entropies[position] = new PositionEntropy(position, entropy(logits, batch, position));
            }
            Arrays.sort(entropies);
            double cumulativeEntropy = 0.0;
            for (PositionEntropy positionEntropy : entropies) {
                cumulativeEntropy += positionEntropy.entropy;
                if (cumulativeEntropy - positionEntropy.entropy <= entropyBound) {
                    int position = positionEntropy.position;
                    accepted[batch][position] = true;
                    result[batch][position] = denoiserCanvas[batch][position];
                }
            }
        }
        this.acceptedTokenMask = accepted;
        return result;
    }

    public int[][] renoiseCanvas(int[][] acceptedCanvas, int curStep) {
        throw new UnsupportedOperationException("DiffusionGemma renoiseCanvas is not implemented yet");
    }

    private double entropy(AbstractTensor logits, int batch, int position) {
        double max = Double.NEGATIVE_INFINITY;
        for (int token = 0; token < vocabSize; token++) {
            max = Math.max(max, logits.get(batch, position, token));
        }
        double sumExp = 0.0;
        double weighted = 0.0;
        for (int token = 0; token < vocabSize; token++) {
            double shifted = logits.get(batch, position, token) - max;
            double exp = FastMath.exp(shifted);
            sumExp += exp;
            weighted += exp * shifted;
        }
        return FastMath.log(sumExp) - (weighted / sumExp);
    }

    private void validateCanvas(int[][] canvas, String name) {
        Objects.requireNonNull(canvas, name);
        if (canvas.length == 0) {
            throw new IllegalArgumentException(name + " must have at least one batch row");
        }
        for (int batch = 0; batch < canvas.length; batch++) {
            if (canvas[batch].length != canvasLength) {
                throw new IllegalArgumentException(name + " row " + batch + " must have canvasLength=" + canvasLength);
            }
        }
    }

    private static int[][] copyCanvas(int[][] source) {
        int[][] copy = new int[source.length][];
        for (int i = 0; i < source.length; i++) {
            copy[i] = Arrays.copyOf(source[i], source[i].length);
        }
        return copy;
    }

    private record PositionEntropy(int position, double entropy) implements Comparable<PositionEntropy> {
        @Override
        public int compareTo(PositionEntropy other) {
            int entropyComparison = Double.compare(this.entropy, other.entropy);
            return entropyComparison != 0 ? entropyComparison : Integer.compare(this.position, other.position);
        }
    }
}
