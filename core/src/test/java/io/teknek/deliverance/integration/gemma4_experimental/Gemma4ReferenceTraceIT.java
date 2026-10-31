package io.teknek.deliverance.integration.gemma4_experimental;

import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.kv.KvCacheSession;
import io.teknek.deliverance.tensor2.TensorRef;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

/** Writes a prefill-only trace for comparison with the NumPy Transformers reference dump. */
class Gemma4ReferenceTraceIT {
    private static final int[] PROMPT = {
            2, 105, 2364, 107, 3048, 735, 531, 5347, 1534, 11691, 5122, 653, 11691,
            10173, 236761, 2900, 776, 611, 3717, 236881, 106, 107, 105, 4368, 107
    };

    @Test
    void writesPrefillTrace() throws IOException {
        Assumptions.assumeTrue(Boolean.getBoolean("deliverance.gemma4.reference.trace"));
        AbstractModel model = Gemma4Suite.getOrCreate();
        Map<String, float[]> trace = new LinkedHashMap<>();
        model.setLayerDebugHook(event -> trace.put(event.layerIndex() + ":" + event.stage(), lastRow(event.hiddenStates())));
        try (KvCacheSession session = model.newKvCacheSession();
             TensorRef ignored = model.batchForwardRef(PROMPT, 0, session)) {
            writeTrace(trace, Path.of(System.getProperty("deliverance.gemma4.reference.trace.path",
                    "/tmp/deliverance-gemma4-reference-trace.txt")));
        } finally {
            model.clearLayerDebugHook();
        }
    }

    private static float[] lastRow(AbstractTensor tensor) {
        int row = tensor.shape().first() - 1;
        float[] result = new float[tensor.shape().last()];
        for (int column = 0; column < result.length; column++) result[column] = tensor.get(row, column);
        return result;
    }

    private static void writeTrace(Map<String, float[]> trace, Path path) throws IOException {
        StringBuilder output = new StringBuilder();
        for (Map.Entry<String, float[]> entry : trace.entrySet()) {
            output.append(entry.getKey());
            for (float value : entry.getValue()) output.append(',').append(value);
            output.append('\n');
        }
        Files.writeString(path, output);
    }
}
