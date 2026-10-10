package io.teknek.deliverance.safetensors;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.impl.BFloat16BufferTensor;
import io.teknek.deliverance.tensor.impl.Float16BufferTensor;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.impl.Q4ByteBufferTensor;
import io.teknek.deliverance.tensor.impl.Q8ByteBufferTensor;
import io.teknek.deliverance.tensor2.TensorRefBackedTensor;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;
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

    @Test
    void mappedQ4ReshapeMatchesLegacyQ4AcrossEveryElement() throws Exception {
        FloatBufferTensor source = new FloatBufferTensor(2, 32);
        for (int row = 0; row < source.shape().first(); row++) {
            for (int column = 0; column < source.shape().last(); column++) {
                source.set((row * 37 + column * 11) % 29 - 14.0f, row, column);
            }
        }
        Q4ByteBufferTensor q4 = new Q4ByteBufferTensor(source);
        SafeTensorWriter.write(tempDir.resolve("model.safetensors"), Map.of(), Map.of("q4.weight", q4));

        try (DefaultWeightLoader loader = new DefaultWeightLoader(tempDir.toFile())) {
            Lighter lighter = new Lighter();
            loader.setLighter(lighter);
            try (AbstractTensor legacy = loader.load("q4.weight");
                 TensorRef mapped = loader.loadRef("q4.weight");
                 TensorRefBackedTensor dense = new TensorRefBackedTensor(
                         lighter.reshape(mapped, io.teknek.deliverance.DType.F32))) {
                for (int row = 0; row < legacy.shape().first(); row++) {
                    for (int column = 0; column < legacy.shape().last(); column++) {
                        assertEquals(legacy.get(row, column), dense.get(row, column), 1.0e-4f,
                                "row=" + row + " column=" + column);
                    }
                }
            }
        }
        q4.close();
        source.close();
    }

    @Test
    void mappedI8ScalarAndReshapeMatchLegacyValuesAndScales() throws Exception {
        FloatBufferTensor source = new FloatBufferTensor(2, 64);
        for (int row = 0; row < source.shape().first(); row++) {
            for (int column = 0; column < source.shape().last(); column++) {
                int block = column / Q8ByteBufferTensor.BLOCK_SIZE;
                source.set((column % Q8ByteBufferTensor.BLOCK_SIZE - 15) * (row + 1) * (block + 1) * 0.25f,
                        row, column);
            }
        }
        Q8ByteBufferTensor q8 = new Q8ByteBufferTensor(source);
        SafeTensorWriter.write(tempDir.resolve("model.safetensors"), Map.of(), Map.of("q8.weight", q8));

        try (DefaultWeightLoader loader = new DefaultWeightLoader(tempDir.toFile())) {
            Lighter lighter = new Lighter();
            loader.setLighter(lighter);
            try (AbstractTensor legacy = loader.load("q8.weight");
                 AbstractTensor legacyScale = loader.load("q8.weight.qb");
                 TensorRef mapped = loader.loadRef("q8.weight");
                 TensorRefBackedTensor dense = new TensorRefBackedTensor(lighter.reshape(mapped, DType.F32))) {
                TensorRef mappedScale = mapped.sidecar("q8.scale");
                for (int row = 0; row < legacy.shape().first(); row++) {
                    for (int block = 0; block < legacyScale.shape().last(); block++) {
                        assertEquals(legacyScale.get(row, block), mappedScale.get(row, block), 0.0f,
                                "scale row=" + row + " block=" + block);
                    }
                    for (int column = 0; column < legacy.shape().last(); column++) {
                        float expected = legacy.get(row, column);
                        assertEquals(expected, mapped.get(row, column), 0.0f,
                                "mapped scalar row=" + row + " column=" + column);
                        assertEquals(expected, dense.get(row, column), 0.0f,
                                "reshaped row=" + row + " column=" + column);
                    }
                }
            }
        }
        q8.close();
        source.close();
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
