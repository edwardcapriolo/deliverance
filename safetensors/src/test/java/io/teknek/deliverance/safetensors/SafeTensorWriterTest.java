package io.teknek.deliverance.safetensors;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.impl.BFloat16BufferTensor;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.impl.Float16BufferTensor;
import io.teknek.deliverance.tensor.impl.Q4ByteBufferTensor;
import io.teknek.deliverance.tensor.impl.Q8ByteBufferTensor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class SafeTensorWriterTest {
    @TempDir
    Path tempDir;

    @Test
    public void writesQ4TensorWithBlockFactors() throws Exception {
        FloatBufferTensor source = new FloatBufferTensor(1, 32);
        for (int i = 0; i < 32; i++) {
            source.set(i - 16, 0, i);
        }
        Q4ByteBufferTensor q4 = new Q4ByteBufferTensor(source);
        Map<String, AbstractTensor> tensors = new LinkedHashMap<>();
        tensors.put("layer.weight", q4);

        Path output = tempDir.resolve("model.safetensors");
        SafeTensorWriter.write(output, Map.of("format", "pt"), tensors);

        try (DefaultWeightLoader loader = new DefaultWeightLoader(tempDir.toFile());
             AbstractTensor tensor = loader.load("layer.weight");
             AbstractTensor block = loader.load("layer.weight.qb")) {
            assertEquals(DType.Q4, loader.tensorInfoMap().get("layer.weight").dType);
            assertTrue(loader.isWeightPresent("layer.weight.qb"));
            assertEquals(1, tensor.shape().first());
            assertEquals(32, tensor.shape().last());
            assertEquals(1, block.shape().last());
        }
    }

    @Test
    public void writesDenseVectorAsRowVector() throws Exception {
        FloatBufferTensor vector = new FloatBufferTensor(4);
        vector.set(1.0f, 0, 0);
        vector.set(2.0f, 0, 1);
        vector.set(3.0f, 0, 2);
        vector.set(4.0f, 0, 3);

        Path output = tempDir.resolve("model.safetensors");
        SafeTensorWriter.write(output, Map.of(), Map.of("norm.weight", vector));

        try (DefaultWeightLoader loader = new DefaultWeightLoader(tempDir.toFile())) {
            assertEquals(2, loader.tensorInfoMap().get("norm.weight").shape.length);
            assertEquals(1, loader.tensorInfoMap().get("norm.weight").shape[0]);
            assertEquals(4, loader.tensorInfoMap().get("norm.weight").shape[1]);
        }
    }

    @Test
    public void writesAndLoadsDense3dTensor() throws Exception {
        FloatBufferTensor tensor3d = new FloatBufferTensor(2, 3, 4);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 4; k++) {
                    tensor3d.set(i * 100 + j * 10 + k, i, j, k);
                }
            }
        }

        Path output = tempDir.resolve("model.safetensors");
        SafeTensorWriter.write(output, Map.of(), Map.of("experts.gate_up_proj", tensor3d));

        try (DefaultWeightLoader loader = new DefaultWeightLoader(tempDir.toFile());
             AbstractTensor loaded = loader.load("experts.gate_up_proj")) {
            assertEquals(3, loader.tensorInfoMap().get("experts.gate_up_proj").shape.length);
            assertEquals(2, loader.tensorInfoMap().get("experts.gate_up_proj").shape[0]);
            assertEquals(3, loader.tensorInfoMap().get("experts.gate_up_proj").shape[1]);
            assertEquals(4, loader.tensorInfoMap().get("experts.gate_up_proj").shape[2]);
            assertEquals(123.0f, loaded.get(1, 2, 3), 0.0f);
        }
    }

    @Test
    public void writesShardedModelWithIndex() throws Exception {
        FloatBufferTensor first = new FloatBufferTensor(1, 32);
        FloatBufferTensor second = new FloatBufferTensor(1, 32);
        for (int i = 0; i < 32; i++) {
            first.set(i - 12, 0, i);
            second.set(i + 3, 0, i);
        }
        Map<String, AbstractTensor> tensors = new LinkedHashMap<>();
        tensors.put("layer1.weight", new Q4ByteBufferTensor(first));
        tensors.put("layer2.weight", new Q4ByteBufferTensor(second));

        SafeTensorWriter.writeModel(tempDir, Map.of("format", "pt"), tensors, 32);

        assertTrue(Files.exists(tempDir.resolve(SafeTensorIndexPojo.MODEL_INDEX_JSON)));
        try (DefaultWeightLoader loader = new DefaultWeightLoader(tempDir.toFile())) {
            assertTrue(loader.isWeightPresent("layer1.weight"));
            assertTrue(loader.isWeightPresent("layer2.weight.qb"));
        }
    }

    @Test
    public void writesQ8TensorWithBlockFactors() {
        FloatBufferTensor source = new FloatBufferTensor(1, 32);
        for (int i = 0; i < 32; i++) {
            source.set(i - 16, 0, i);
        }
        Q8ByteBufferTensor q8 = new Q8ByteBufferTensor(source);
        Path output = tempDir.resolve("model.safetensors");
        SafeTensorWriter.write(output, Map.of(), Map.of("layer.weight", q8));

        try (DefaultWeightLoader loader = new DefaultWeightLoader(tempDir.toFile());
             AbstractTensor loaded = loader.load("layer.weight");
             AbstractTensor block = loader.load("layer.weight.qb")) {
            assertEquals(DType.I8, loader.tensorInfoMap().get("layer.weight").dType);
            assertEquals(DType.F32, loader.tensorInfoMap().get("layer.weight.qb").dType);
            assertEquals(32, loaded.shape().last());
            assertEquals(1, block.shape().last());
            assertEquals(source.get(0, 7), loaded.get(0, 7), 0.2f);
        }
        source.close();
    }

    @Test
    public void writesAndLoadsBfloat16AndFloat16Tensors() {
        BFloat16BufferTensor bf16 = new BFloat16BufferTensor(1, 2);
        bf16.set(1.25f, 0, 0);
        bf16.set(-2.5f, 0, 1);
        Float16BufferTensor f16 = new Float16BufferTensor(1, 2);
        f16.set(3.5f, 0, 0);
        f16.set(-4.5f, 0, 1);
        SafeTensorWriter.write(tempDir.resolve("model.safetensors"), Map.of(), Map.of(
                "bf16.weight", bf16, "f16.weight", f16));

        try (DefaultWeightLoader loader = new DefaultWeightLoader(tempDir.toFile());
             AbstractTensor loadedBf16 = loader.load("bf16.weight");
            AbstractTensor loadedF16 = loader.load("f16.weight")) {
            assertEquals(DType.BF16, loadedBf16.dType());
            assertEquals(DType.F16, loader.tensorInfoMap().get("f16.weight").dType);
            assertEquals(DType.F32, loadedF16.dType());
            assertEquals(1.25f, loadedBf16.get(0, 0), 0.02f);
            assertEquals(-4.5f, loadedF16.get(0, 1), 0.01f);
        }
        bf16.close();
        f16.close();
    }
}
