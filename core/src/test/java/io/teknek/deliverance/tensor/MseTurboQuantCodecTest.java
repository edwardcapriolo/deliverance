package io.teknek.deliverance.tensor;

import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class MseTurboQuantCodecTest {

    @Test
    void nextPowerOfTwoRoundsUp() {
        assertEquals(1, MseTurboQuantCodec.nextPowerOfTwo(1));
        assertEquals(8, MseTurboQuantCodec.nextPowerOfTwo(8));
        assertEquals(16, MseTurboQuantCodec.nextPowerOfTwo(9));
        assertThrows(IllegalArgumentException.class, () -> MseTurboQuantCodec.nextPowerOfTwo(0));
    }

    @Test
    void hadamardKnownLengthFourTransform() {
        float[] values = {1, 2, 3, 4};

        MseTurboQuantCodec.fastWalshHadamard(values);

        assertArrayEquals(new float[]{10, -2, -4, 0}, values, 0.0f);
    }

    @Test
    void hadamardAppliedTwiceReturnsLengthTimesInput() {
        float[] values = {1.5f, -2.0f, 3.25f, 4.5f, -5.0f, 6.0f, 7.0f, -8.0f};
        float[] original = values.clone();

        MseTurboQuantCodec.fastWalshHadamard(values);
        MseTurboQuantCodec.fastWalshHadamard(values);

        for (int i = 0; i < values.length; i++) {
            assertEquals(original[i] * values.length, values[i], 0.0001f, "index=" + i);
        }
    }

    @Test
    void hadamardRejectsNonPowerOfTwoLength() {
        assertThrows(IllegalArgumentException.class,
                () -> MseTurboQuantCodec.fastWalshHadamard(new float[]{1, 2, 3}));
    }

    @Test
    void packAndUnpackCodesAcrossByteBoundaries() {
        for (int bitWidth = 1; bitWidth <= 8; bitWidth++) {
            int mask = (1 << bitWidth) - 1;
            byte[] packed = new byte[(21 * bitWidth + 7) / 8];
            for (int i = 0; i < 21; i++) {
                MseTurboQuantCodec.packCode(packed, i, bitWidth, (i * 3) & mask);
            }
            for (int i = 0; i < 21; i++) {
                assertEquals((i * 3) & mask, MseTurboQuantCodec.unpackCode(packed, i, bitWidth),
                        "bitWidth=" + bitWidth + " index=" + i);
            }
        }
    }

    @Test
    void codebookIsSortedAndNearestIndexIsStable() {
        float[] codebook = MseTurboQuantCodec.codebook(4);

        assertEquals(16, codebook.length);
        for (int i = 1; i < codebook.length; i++) {
            assertTrue(codebook[i - 1] < codebook[i], "codebook must be strictly sorted");
        }
        assertEquals(0, MseTurboQuantCodec.nearestCodebookIndex(-100.0f, codebook));
        assertEquals(codebook.length - 1, MseTurboQuantCodec.nearestCodebookIndex(100.0f, codebook));
    }

    @Test
    void thresholdQuantizerMatchesNearestCentroidScan() {
        MseTurboQuantCodec.ScalarQuantizer quantizer = MseTurboQuantCodec.quantizer(4);
        for (float value = -5.0f; value <= 5.0f; value += 0.03125f) {
            assertEquals(
                    MseTurboQuantCodec.nearestCodebookIndex(value, quantizer.codebook()),
                    MseTurboQuantCodec.quantizeScalar(value, quantizer),
                    "value=" + value);
        }
        for (int i = 0; i < quantizer.thresholds().length; i++) {
            assertEquals(i, MseTurboQuantCodec.quantizeScalar(quantizer.thresholds()[i], quantizer),
                    "threshold tie should choose lower code");
        }
    }

    @Test
    void rotationSignsAreCachedAndDeterministic() {
        float[] signs = MseTurboQuantCodec.rotationSigns(16);

        assertTrue(signs == MseTurboQuantCodec.rotationSigns(16));
        for (int i = 0; i < signs.length; i++) {
            assertEquals(MseTurboQuantCodec.rotationSign(i), signs[i], 0.0f);
        }
    }

    @Test
    void encodeDecodeRowProducesOwnedApproximationAndShrinksPayload() {
        try (FloatBufferTensor source = new FloatBufferTensor(1, 8);
             FloatBufferTensor restored = new FloatBufferTensor(1, 8)) {
            for (int i = 0; i < 8; i++) {
                source.set((i + 1) * (i % 2 == 0 ? 1.0f : -1.0f), 0, i);
            }
            MseTurboQuantCodec.EncodedRows encoded = MseTurboQuantCodec.allocate(1, 8, 4);
            MseTurboQuantCodec.Scratch scratch = new MseTurboQuantCodec.Scratch(encoded.rotatedDim());

            assertEquals(1, MseTurboQuantCodec.encodeRow(source, encoded, 0, null, scratch));
            assertEquals(1, MseTurboQuantCodec.decodeRow(encoded, restored, 0, null, scratch));

            double squaredError = 0.0;
            for (int i = 0; i < 8; i++) {
                double diff = source.get(0, i) - restored.get(0, i);
                squaredError += diff * diff;
            }
            double rmse = Math.sqrt(squaredError / 8.0);
            assertTrue(rmse < 2.5, "rmse=" + rmse);
            assertTrue(encoded.encodedBytes() < 8L * Float.BYTES);
        }
    }
}
