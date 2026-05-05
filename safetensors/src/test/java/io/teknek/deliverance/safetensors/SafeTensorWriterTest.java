package io.teknek.deliverance.safetensors;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.impl.Q4ByteBufferTensor;
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
}
