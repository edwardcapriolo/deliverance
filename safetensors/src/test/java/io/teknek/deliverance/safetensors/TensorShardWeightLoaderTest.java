package io.teknek.deliverance.safetensors;

import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorDisplayUtil;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class TensorShardWeightLoaderTest {
    @TempDir
    Path tempDir;

    @Test
    public void loadsRowShardFromDenseTensor() {
        writeMatrix();

        try (DefaultWeightLoader loader = new DefaultWeightLoader(tempDir.toFile());
             AbstractTensor shard = loader.load("layer.weight", new TensorShardSpec(TensorShardAxis.ROWS, 2, 4))) {
            assertEquals("""
                    [0][0]= 12.0000 [0][1]= 13.0000 [0][2]= 14.0000 [0][3]= 15.0000 [0][4]= 16.0000 [0][5]= 17.0000
                    [1][0]= 18.0000 [1][1]= 19.0000 [1][2]= 20.0000 [1][3]= 21.0000 [1][4]= 22.0000 [1][5]= 23.0000
                    """.trim(), normalize(TensorDisplayUtil.pretty2dDisplayAll(shard)));
        }
    }

    @Test
    public void loadsColumnShardFromDenseTensor() {
        writeMatrix();

        try (DefaultWeightLoader loader = new DefaultWeightLoader(tempDir.toFile());
             AbstractTensor shard = loader.load("layer.weight", new TensorShardSpec(TensorShardAxis.COLUMNS, 2, 4))) {
            assertEquals("""
                    [0][0]=  2.0000 [0][1]=  3.0000
                    [1][0]=  8.0000 [1][1]=  9.0000
                    [2][0]= 14.0000 [2][1]= 15.0000
                    [3][0]= 20.0000 [3][1]= 21.0000
                    """.trim(), normalize(TensorDisplayUtil.pretty2dDisplayAll(shard)));
        }
    }

    @Test
    public void rejectsInvalidShardRanges() {
        writeMatrix();

        try (DefaultWeightLoader loader = new DefaultWeightLoader(tempDir.toFile())) {
            assertThrows(IllegalArgumentException.class,
                    () -> loader.load("layer.weight", new TensorShardSpec(TensorShardAxis.ROWS, 3, 5)));
            assertThrows(IllegalArgumentException.class,
                    () -> loader.load("layer.weight", new TensorShardSpec(TensorShardAxis.COLUMNS, 5, 7)));
        }
    }

    private void writeMatrix() {
        FloatBufferTensor matrix = new FloatBufferTensor(4, 6);
        int value = 0;
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 6; col++) {
                matrix.set(value++, row, col);
            }
        }
        SafeTensorWriter.write(tempDir.resolve("model.safetensors"), Map.of(), Map.of("layer.weight", matrix));
    }

    private static String normalize(String display) {
        return display.strip().replaceAll("(?m) +$", "");
    }
}
