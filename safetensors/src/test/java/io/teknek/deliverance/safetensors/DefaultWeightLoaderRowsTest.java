package io.teknek.deliverance.safetensors;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorInfo;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.lang.reflect.Field;
import java.nio.file.Path;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class DefaultWeightLoaderRowsTest {
    @TempDir
    Path tempDir;

    @Test
    public void loadsRowsFromUnsplitTensor() throws Exception {
        FloatBufferTensor source = matrix(4, 4, 0);
        SafeTensorWriter.write(tempDir.resolve("model.safetensors"), Map.of(), Map.of("plain.weight", source));

        try (DefaultWeightLoader loader = new DefaultWeightLoader(tempDir.toFile());
             AbstractTensor rows = loader.loadRows("plain.weight", 1, 2)) {
            assertEquals(2, rows.shape().first());
            assertEquals(4, rows.shape().last());
            assertEquals(4.0f, rows.get(0, 0));
            assertEquals(7.0f, rows.get(0, 3));
            assertEquals(8.0f, rows.get(1, 0));
            assertEquals(11.0f, rows.get(1, 3));
        }
    }

    @Test
    public void loadsRowsFromLogicalSplitTensor() throws Exception {
        FloatBufferTensor part0 = matrix(2, 4, 0);
        FloatBufferTensor part1 = matrix(3, 4, 8);

        SafeTensorWriter.writeShardFile(
                tempDir.resolve("model-00001-of-00002.safetensors"),
                Map.of(),
                SafeTensorWriter.flatten(Map.of("logical.weight-part-0", part0))
        );
        SafeTensorWriter.writeShardFile(
                tempDir.resolve("model-00002-of-00002.safetensors"),
                Map.of(),
                SafeTensorWriter.flatten(Map.of("logical.weight-part-1", part1))
        );

        JsonUtils.om.writeValue(tempDir.resolve(SafeTensorIndexPojo.MODEL_INDEX_JSON).toFile(),
                new SafeTensorIndexPojo(
                        Map.of(),
                        Map.of(
                                "logical.weight-part-0", "model-00001-of-00002.safetensors",
                                "logical.weight-part-1", "model-00002-of-00002.safetensors"
                        )
                ));

        try (DefaultWeightLoader loader = new DefaultWeightLoader(tempDir.toFile())) {
            tensorInfoMap(loader).put("logical.weight",
                    new TensorInfo(DType.F32, new long[]{5, 4}, new long[]{0, 80}));

            try (AbstractTensor rows = loader.loadRows("logical.weight", 1, 3)) {
                assertEquals(3, rows.shape().first());
                assertEquals(4, rows.shape().last());
                assertEquals(4.0f, rows.get(0, 0));
                assertEquals(7.0f, rows.get(0, 3));
                assertEquals(8.0f, rows.get(1, 0));
                assertEquals(11.0f, rows.get(1, 3));
                assertEquals(12.0f, rows.get(2, 0));
                assertEquals(15.0f, rows.get(2, 3));
            }
        }
    }

    private static FloatBufferTensor matrix(int rows, int cols, int start) {
        FloatBufferTensor tensor = new FloatBufferTensor(rows, cols);
        int value = start;
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                tensor.set(value++, row, col);
            }
        }
        return tensor;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, TensorInfo> tensorInfoMap(DefaultWeightLoader loader) throws Exception {
        Field field = DefaultWeightLoader.class.getDeclaredField("allTensorInfoMap");
        field.setAccessible(true);
        return (Map<String, TensorInfo>) field.get(loader);
    }
}
