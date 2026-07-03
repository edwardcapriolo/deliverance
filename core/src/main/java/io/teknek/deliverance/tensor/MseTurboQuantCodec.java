package io.teknek.deliverance.tensor;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.model.InferenceProfiler;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * MSE-oriented TurboQuant-style row-vector codec used by prefix-cache snapshot storage.
 *
 * <p>This is the reconstruction-oriented part of the TurboQuant paper: normalize a vector, apply a deterministic
 * sign-Hadamard rotation, quantize rotated coordinates with a Lloyd-Max scalar codebook, and reconstruct by reversing
 * those steps. It does not implement the QJL residual used by the paper's inner-product-optimal variant.</p>
 *
 * <p>The class is package-private on purpose. Prefix-cache code owns storage/lifecycle, while this class owns the numeric
 * mechanics so they can be directly unit tested.</p>
 */
final class MseTurboQuantCodec {
    static final long ROTATION_SEED = 0x6A09E667F3BCC909L;
    /**
     * JVM-wide cache of Lloyd-Max codebooks keyed by bit width.
     *
     * <p>The domain is bounded by settings validation to bit widths 1 through 8, so this cache can hold at most eight
     * entries and is shared across all users/sessions in the process. Codebooks are treated as immutable after creation;
     * callers must not mutate the returned arrays because that would affect every concurrent user of the same bit width.</p>
     *
     * <p>The synchronized map prevents concurrent structural corruption. A future hardening pass should replace this with
     * an explicitly bounded immutable/fixed-array cache to make the production concurrency contract more obvious.</p>
     */
    private static final Map<Integer, ScalarQuantizer> QUANTIZERS = Collections.synchronizedMap(new HashMap<>());
    /** JVM-wide cache of deterministic sign vectors keyed by rotated dimension. Entries are read-only after creation. */
    private static final Map<Integer, float[]> ROTATION_SIGNS = Collections.synchronizedMap(new HashMap<>());

    private MseTurboQuantCodec() {
    }

    /** Encoded row block for a fixed row count, row width, and bit width. */
    record EncodedRows(byte[] packedCodes, float[] norms, int bitWidth, int kvLength, int rotatedDim) {
        long encodedBytes() {
            return packedCodes.length + (long) norms.length * Float.BYTES;
        }
    }

    /** Sorted scalar centroids plus midpoint thresholds between adjacent centroids. */
    record ScalarQuantizer(float[] codebook, float[] thresholds) {
    }

    /** Reusable per-thread/per-call scratch space for encode/decode row transforms. */
    static final class Scratch {
        final float[] rotated;

        Scratch(int rotatedDim) {
            this.rotated = new float[rotatedDim];
        }
    }

    /** Allocates a packed-code/norm container for {@code rows} vectors of length {@code kvLength}. */
    static EncodedRows allocate(int rows, int kvLength, int bitWidth) {
        if (rows < 0) {
            throw new IllegalArgumentException("rows must be >= 0");
        }
        if (kvLength < 1) {
            throw new IllegalArgumentException("kvLength must be positive");
        }
        if (bitWidth < 1 || bitWidth > 8) {
            throw new IllegalArgumentException("bitWidth must be between 1 and 8");
        }
        int rotatedDim = nextPowerOfTwo(kvLength);
        long totalCodes = Math.multiplyExact((long) rows, rotatedDim);
        byte[] packedCodes = new byte[Math.toIntExact((totalCodes * bitWidth + 7) / 8)];
        float[] norms = new float[rows];
        return new EncodedRows(packedCodes, norms, bitWidth, kvLength, rotatedDim);
    }

    /**
     * Encodes one readable row into {@code encoded} at {@code rowIndex}.
     *
     * <p>The row is borrowed; this method does not close it.</p>
     */
    static int encodeRow(ReadableTensor row, EncodedRows encoded, int rowIndex) {
        return encodeRow(row, encoded, rowIndex, null, new Scratch(encoded.rotatedDim()));
    }

    /**
     * Encodes one readable row and records optional substep metrics when {@code metricRegistry} is non-null.
     */
    static int encodeRow(ReadableTensor row, EncodedRows encoded, int rowIndex, MetricRegistry metricRegistry) {
        return encodeRow(row, encoded, rowIndex, metricRegistry, new Scratch(encoded.rotatedDim()));
    }

    /** Encodes one readable row using caller-provided scratch storage. */
    static int encodeRow(ReadableTensor row, EncodedRows encoded, int rowIndex, MetricRegistry metricRegistry,
            Scratch scratch) {
        ScalarQuantizer quantizer = quantizer(encoded.bitWidth());
        float[] signs = rotationSigns(encoded.rotatedDim());
        float invSqrtDim = (float) (1.0 / Math.sqrt(encoded.rotatedDim()));
        long normStart = System.nanoTime();
        float normSquared = 0.0f;
        for (int i = 0; i < encoded.kvLength(); i++) {
            float value = row.get(0, i);
            normSquared += value * value;
        }
        float norm = (float) Math.sqrt(normSquared);
        recordTimer(metricRegistry, "kvbuffercache.prefix.turboquant.row.norm", normStart);
        encoded.norms()[rowIndex] = norm;
        float[] rotated = scratch.rotated;
        if (norm != 0.0f) {
            long signStart = System.nanoTime();
            float inverseNorm = 1.0f / norm;
            for (int i = 0; i < encoded.kvLength(); i++) {
                rotated[i] = row.get(0, i) * inverseNorm * signs[i];
            }
            if (encoded.rotatedDim() > encoded.kvLength()) {
                Arrays.fill(rotated, encoded.kvLength(), encoded.rotatedDim(), 0.0f);
            }
            recordTimer(metricRegistry, "kvbuffercache.prefix.turboquant.row.sign", signStart);
            long rotateStart = System.nanoTime();
            fastWalshHadamard(rotated);
            for (int i = 0; i < encoded.rotatedDim(); i++) {
                rotated[i] *= invSqrtDim;
            }
            recordTimer(metricRegistry, "kvbuffercache.prefix.turboquant.row.hadamard", rotateStart);
        }
        long baseCode = (long) rowIndex * encoded.rotatedDim();
        float coordinateScale = (float) Math.sqrt(encoded.rotatedDim());
        long quantizeStart = System.nanoTime();
        for (int i = 0; i < encoded.rotatedDim(); i++) {
            int code = quantizeScalar(rotated[i] * coordinateScale, quantizer);
            packCode(encoded.packedCodes(), baseCode + i, encoded.bitWidth(), code);
        }
        recordTimer(metricRegistry, "kvbuffercache.prefix.turboquant.row.quantize_pack", quantizeStart);
        recordCounter(metricRegistry, "kvbuffercache.prefix.turboquant.row.count", 1);
        recordCounter(metricRegistry, "kvbuffercache.prefix.turboquant.coordinate.count", encoded.rotatedDim());
        return rowIndex + 1;
    }

    /**
     * Decodes one row from {@code encoded} into {@code row}.
     *
     * <p>The destination row is borrowed; this method does not close it.</p>
     */
    static int decodeRow(EncodedRows encoded, AbstractTensor row, int rowIndex) {
        return decodeRow(encoded, row, rowIndex, null, new Scratch(encoded.rotatedDim()));
    }

    /**
     * Decodes one row and records optional substep metrics when {@code metricRegistry} is non-null.
     */
    static int decodeRow(EncodedRows encoded, AbstractTensor row, int rowIndex, MetricRegistry metricRegistry) {
        return decodeRow(encoded, row, rowIndex, metricRegistry, new Scratch(encoded.rotatedDim()));
    }

    /** Decodes one row using caller-provided scratch storage. */
    static int decodeRow(EncodedRows encoded, AbstractTensor row, int rowIndex, MetricRegistry metricRegistry,
            Scratch scratch) {
        float[] codebook = codebook(encoded.bitWidth());
        float[] signs = rotationSigns(encoded.rotatedDim());
        float invSqrtDim = (float) (1.0 / Math.sqrt(encoded.rotatedDim()));
        float[] rotated = scratch.rotated;
        long baseCode = (long) rowIndex * encoded.rotatedDim();
        long unpackStart = System.nanoTime();
        for (int i = 0; i < encoded.rotatedDim(); i++) {
            int code = unpackCode(encoded.packedCodes(), baseCode + i, encoded.bitWidth());
            rotated[i] = codebook[code] * invSqrtDim;
        }
        recordTimer(metricRegistry, "kvbuffercache.prefix.turboquant.row.unpack", unpackStart);
        long rotateStart = System.nanoTime();
        fastWalshHadamard(rotated);
        recordTimer(metricRegistry, "kvbuffercache.prefix.turboquant.row.inverse_hadamard", rotateStart);
        float normScale = encoded.norms()[rowIndex] * invSqrtDim;
        long writeStart = System.nanoTime();
        for (int i = 0; i < encoded.kvLength(); i++) {
            row.set(rotated[i] * normScale * signs[i], 0, i);
        }
        recordTimer(metricRegistry, "kvbuffercache.prefix.turboquant.row.write", writeStart);
        recordCounter(metricRegistry, "kvbuffercache.prefix.turboquant.decode.row.count", 1);
        return rowIndex + 1;
    }

    private static void recordTimer(MetricRegistry metricRegistry, String name, long startNanos) {
        if (metricRegistry != null) {
            InferenceProfiler.timer(metricRegistry, name).update(System.nanoTime() - startNanos, TimeUnit.NANOSECONDS);
        }
    }

    private static void recordCounter(MetricRegistry metricRegistry, String name, long amount) {
        if (metricRegistry != null) {
            InferenceProfiler.counter(metricRegistry, name).inc(amount);
        }
    }

    /** Returns the next power of two greater than or equal to {@code value}. */
    static int nextPowerOfTwo(int value) {
        if (value < 1) {
            throw new IllegalArgumentException("value must be positive");
        }
        int highest = Integer.highestOneBit(value);
        return highest == value ? value : highest << 1;
    }

    /** Deterministic pseudo-random sign used before the Hadamard rotation. */
    static float rotationSign(int index) {
        long x = ROTATION_SEED + 0x9E3779B97F4A7C15L * index;
        x = (x ^ (x >>> 30)) * 0xBF58476D1CE4E5B9L;
        x = (x ^ (x >>> 27)) * 0x94D049BB133111EBL;
        x = x ^ (x >>> 31);
        return (x & 1L) == 0L ? 1.0f : -1.0f;
    }

    /** Returns deterministic rotation signs for coordinates {@code 0..length-1}. */
    static float[] rotationSigns(int length) {
        return ROTATION_SIGNS.computeIfAbsent(length, ignored -> {
            float[] signs = new float[length];
            for (int i = 0; i < length; i++) {
                signs[i] = rotationSign(i);
            }
            return signs;
        });
    }

    /**
     * In-place unnormalized Walsh-Hadamard transform.
     *
     * <p>For power-of-two length {@code n}, applying this method twice yields {@code n * original}. Callers normalize by
     * {@code 1/sqrt(n)} when they need an orthonormal rotation.</p>
     */
    static void fastWalshHadamard(float[] values) {
        if (Integer.bitCount(values.length) != 1) {
            throw new IllegalArgumentException("Hadamard length must be a power of two");
        }
        for (int step = 1; step < values.length; step <<= 1) {
            for (int base = 0; base < values.length; base += step << 1) {
                for (int i = 0; i < step; i++) {
                    float a = values[base + i];
                    float b = values[base + i + step];
                    values[base + i] = a + b;
                    values[base + i + step] = a - b;
                }
            }
        }
    }

    /** Returns a sorted Lloyd-Max codebook for a standard normal approximation. */
    static float[] codebook(int bitWidth) {
        return quantizer(bitWidth).codebook();
    }

    /** Returns the scalar quantizer for {@code bitWidth}, including midpoint thresholds. */
    static ScalarQuantizer quantizer(int bitWidth) {
        return QUANTIZERS.computeIfAbsent(bitWidth, ignored -> {
            float[] codebook = buildNormalLloydMaxCodebook(bitWidth);
            float[] thresholds = new float[Math.max(0, codebook.length - 1)];
            for (int i = 0; i < thresholds.length; i++) {
                thresholds[i] = (codebook[i] + codebook[i + 1]) * 0.5f;
            }
            return new ScalarQuantizer(codebook, thresholds);
        });
    }

    /** Builds a scalar Lloyd-Max codebook for a standard normal approximation over [-6, 6]. */
    static float[] buildNormalLloydMaxCodebook(int bitWidth) {
        int levels = 1 << bitWidth;
        float[] centroids = new float[levels];
        float min = -6.0f;
        float max = 6.0f;
        for (int i = 0; i < levels; i++) {
            centroids[i] = levels == 1 ? 0.0f : min + (max - min) * (i + 0.5f) / levels;
        }
        int samples = 20_001;
        float[] sampleValues = new float[samples];
        float[] sampleWeights = new float[samples];
        float dx = (max - min) / (samples - 1);
        for (int i = 0; i < samples; i++) {
            float x = min + i * dx;
            sampleValues[i] = x;
            sampleWeights[i] = (float) Math.exp(-0.5f * x * x);
        }
        for (int iter = 0; iter < 80; iter++) {
            float[] weightedSums = new float[levels];
            float[] weights = new float[levels];
            for (int i = 0; i < samples; i++) {
                int nearest = nearestCodebookIndex(sampleValues[i], centroids);
                weightedSums[nearest] += sampleValues[i] * sampleWeights[i];
                weights[nearest] += sampleWeights[i];
            }
            for (int i = 0; i < levels; i++) {
                if (weights[i] > 0.0f) {
                    centroids[i] = weightedSums[i] / weights[i];
                }
            }
            Arrays.sort(centroids);
        }
        return centroids;
    }

    /** Returns the nearest centroid index in {@code codebook}. */
    static int nearestCodebookIndex(float value, float[] codebook) {
        int best = 0;
        float bestDistance = Math.abs(value - codebook[0]);
        for (int i = 1; i < codebook.length; i++) {
            float distance = Math.abs(value - codebook[i]);
            if (distance < bestDistance) {
                bestDistance = distance;
                best = i;
            }
        }
        return best;
    }

    /** Quantizes a scalar using midpoint thresholds. Exact threshold ties choose the lower code. */
    static int quantizeScalar(float value, ScalarQuantizer quantizer) {
        float[] thresholds = quantizer.thresholds();
        for (int i = 0; i < thresholds.length; i++) {
            if (value <= thresholds[i]) {
                return i;
            }
        }
        return thresholds.length;
    }

    /** Packs one fixed-width code into {@code packed}. */
    static void packCode(byte[] packed, long codeIndex, int bitWidth, int code) {
        if (bitWidth == 4) {
            int byteIndex = Math.toIntExact(codeIndex >>> 1);
            if ((codeIndex & 1L) == 0L) {
                packed[byteIndex] = (byte) (packed[byteIndex] | (code & 0x0F));
            } else {
                packed[byteIndex] = (byte) (packed[byteIndex] | ((code & 0x0F) << 4));
            }
            return;
        }
        if (bitWidth == 8) {
            packed[Math.toIntExact(codeIndex)] = (byte) code;
            return;
        }
        long bitOffset = codeIndex * bitWidth;
        for (int bit = 0; bit < bitWidth; bit++) {
            if (((code >>> bit) & 1) != 0) {
                long absoluteBit = bitOffset + bit;
                int byteIndex = Math.toIntExact(absoluteBit >>> 3);
                packed[byteIndex] = (byte) (packed[byteIndex] | (1 << (absoluteBit & 7)));
            }
        }
    }

    /** Unpacks one fixed-width code from {@code packed}. */
    static int unpackCode(byte[] packed, long codeIndex, int bitWidth) {
        if (bitWidth == 4) {
            int value = packed[Math.toIntExact(codeIndex >>> 1)] & 0xFF;
            return (codeIndex & 1L) == 0L ? value & 0x0F : (value >>> 4) & 0x0F;
        }
        if (bitWidth == 8) {
            return packed[Math.toIntExact(codeIndex)] & 0xFF;
        }
        long bitOffset = codeIndex * bitWidth;
        int code = 0;
        for (int bit = 0; bit < bitWidth; bit++) {
            long absoluteBit = bitOffset + bit;
            int byteIndex = Math.toIntExact(absoluteBit >>> 3);
            if (((packed[byteIndex] >>> (absoluteBit & 7)) & 1) != 0) {
                code |= 1 << bit;
            }
        }
        return code;
    }
}
