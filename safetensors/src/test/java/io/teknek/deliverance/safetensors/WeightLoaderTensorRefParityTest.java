package io.teknek.deliverance.safetensors;

import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.impl.BFloat16BufferTensor;
import io.teknek.deliverance.tensor.impl.Float16BufferTensor;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.impl.Q4ByteBufferTensor;
import io.teknek.deliverance.tensor2.TensorRefBackedTensor;
import io.teknek.deliverance.tensor2.Lighter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

class WeightLoaderTensorRefParityTest {
    @TempDir
    Path tempDir;

    @Test
    void loadRefMatchesLoadAcrossDenseAndQuantizedDTypes() throws Exception {
        FloatBufferTensor f32 = new FloatBufferTensor(2, 3);
        f32.set(1.0f, 0, 0);
        f32.set(-2.5f, 1, 2);

        Float16BufferTensor f16 = new Float16BufferTensor(1, 4);
        f16.set(0.5f, 0, 0);
        f16.set(-3.25f, 0, 3);

        BFloat16BufferTensor bf16 = new BFloat16BufferTensor(1, 4);
        bf16.set(0.75f, 0, 0);
        bf16.set(-4.5f, 0, 3);

        FloatBufferTensor q4Source = new FloatBufferTensor(1, 32);
        for (int column = 0; column < 32; column++) {
            q4Source.set(column - 16.0f, 0, column);
        }
        Q4ByteBufferTensor q4 = new Q4ByteBufferTensor(q4Source);

        Map<String, AbstractTensor> tensors = new LinkedHashMap<>();
        tensors.put("f32.weight", f32);
        tensors.put("f16.weight", f16);
        tensors.put("bf16.weight", bf16);
        tensors.put("q4.weight", q4);
        SafeTensorWriter.write(tempDir.resolve("model.safetensors"), Map.of(), tensors);

        try (DefaultWeightLoader loader = new DefaultWeightLoader(tempDir.toFile())) {
            loader.setLighter(new Lighter());
            assertParity(loader, "f32.weight", 0, 0, 1.0f, "F32");
            assertParity(loader, "f16.weight", 0, 3, -3.25f, "F16");
            assertParity(loader, "bf16.weight", 0, 3, -4.5f, "BF16");
            assertParity(loader, "q4.weight", 0, 16, 0.0f, "Q4");
        }
    }

    private static void assertParity(DefaultWeightLoader loader, String name, int row, int column,
            float expected, String refDType) {
        try (AbstractTensor legacy = loader.load(name);
             TensorRefBackedTensor refView = new TensorRefBackedTensor(loader.loadRef(name))) {
            assertEquals(legacy.shape(), refView.shape(), name);
            assertEquals(refDType, refView.dType().name(), name);
            assertEquals(legacy.get(row, column), refView.get(row, column), 0.0f, name);
            assertEquals(expected, refView.get(row, column), 0.1f, name);
        }
    }
}
