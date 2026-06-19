package io.teknek.deliverance.model.hf;

import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.tensor.AbstractTensor;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Adapter used by Deliverance model tests that port Hugging Face text-model common tests.
 *
 * <p>HF model tests use Python mixins. Deliverance uses JUnit 5 interface default methods, while each
 * model supplies its own tiny checkpoint writer and loader.</p>
 */
public interface HfTextCommonTestAdapter {
    Path hfTestTempDir();

    Path writeTinyCheckpoint(String name, int seed);

    AbstractModel loadTinyModel(Path modelDir);

    Config loadTinyConfig(Path modelDir);

    Config roundTripConfig(Config config) throws Exception;

    int[] hfSampleTokenIds();

    AbstractTensor makeInputsEmbeds(int rows, int embeddingLength, int seed);

    default void assertModelSpecificConfigRoundTrip(Config expected, Config actual) {
    }

    default void closeModel(AbstractModel model) {
        if (model instanceof AutoCloseable closeable) {
            try {
                closeable.close();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
    }

    default void assertFiniteTensor(AbstractTensor tensor) {
        for (int row = 0; row < tensor.shape().first(); row++) {
            for (int col = 0; col < tensor.shape().last(); col++) {
                float value = tensor.get(row, col);
                assertTrue(Float.isFinite(value), "non-finite value row=" + row + " col=" + col + " value=" + value);
            }
        }
    }

    default void assertTensorsClose(AbstractTensor expected, AbstractTensor actual, float tolerance, String message) {
        assertEquals(expected.shape().first(), actual.shape().first(), message + " rows");
        assertEquals(expected.shape().last(), actual.shape().last(), message + " cols");
        float maxAbs = 0.0f;
        for (int row = 0; row < expected.shape().first(); row++) {
            for (int col = 0; col < expected.shape().last(); col++) {
                maxAbs = Math.max(maxAbs, Math.abs(expected.get(row, col) - actual.get(row, col)));
            }
        }
        assertTrue(maxAbs <= tolerance, message + " maxAbs=" + maxAbs + " tolerance=" + tolerance);
    }
}
