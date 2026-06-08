package io.teknek.deliverance.safetensors;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorDisplayUtil;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.impl.Q4ByteBufferTensor;
import io.teknek.deliverance.tensor.operations.MachineSpec;
import io.teknek.deliverance.tensor.operations.PanamaTensorOperations;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.mockito.Mockito;

import java.nio.file.Path;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class TensorShardWeightLoaderTest {
    @TempDir
    Path tempDir;

    @Test
    public void loadsRowShardAsLocalDenseTensor() {
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
    public void loadsColumnShardAsLocalDenseTensor() {
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

    @Test
    public void loadsQ4RowShardAsLocalQ4Tensor() {
        writeQ4Matrix();

        try (DefaultWeightLoader loader = new DefaultWeightLoader(tempDir.toFile());
             AbstractTensor full = loader.load("q4.weight");
             AbstractTensor shard = loader.load("q4.weight", new TensorShardSpec(TensorShardAxis.ROWS, 1, 3))) {
            assertEquals(DType.Q4, shard.dType());
            assertEquals(2, shard.shape().first());
            assertEquals(64, shard.shape().last());
            assertQ4ShardEquals(full, shard, 1, 0);
        }
    }

    @Test
    public void loadsWideQ4RowShardUsingLocalPackedLayout() {
        writeWideQ4Matrix();

        try (DefaultWeightLoader loader = new DefaultWeightLoader(tempDir.toFile());
             AbstractTensor full = loader.load("wide_q4.weight");
             AbstractTensor shard = loader.load("wide_q4.weight", new TensorShardSpec(TensorShardAxis.ROWS, 96, 128))) {
            assertEquals(DType.Q4, shard.dType());
            assertEquals(32, shard.shape().first());
            assertEquals(2304, shard.shape().last());
            assertQ4ShardEquals(full, shard, 96, 0);
        }
    }

    @Test
    public void loadsWideQ4RowShardFromIndexedShardFile() {
        writeWideQ4IndexedModel();

        try (DefaultWeightLoader loader = new DefaultWeightLoader(tempDir.toFile());
             AbstractTensor full = loader.load("wide_q4.weight");
             AbstractTensor shard = loader.load("wide_q4.weight", new TensorShardSpec(TensorShardAxis.ROWS, 96, 128))) {
            assertEquals(DType.Q4, shard.dType());
            assertEquals(32, shard.shape().first());
            assertEquals(2304, shard.shape().last());
            assertQ4ShardEquals(full, shard, 96, 0);
        }
    }

    @Test
    public void loadsQ4ColumnShardAsLocalQ4Tensor() {
        writeQ4Matrix();

        try (DefaultWeightLoader loader = new DefaultWeightLoader(tempDir.toFile());
             AbstractTensor full = loader.load("q4.weight");
             AbstractTensor shard = loader.load("q4.weight", new TensorShardSpec(TensorShardAxis.COLUMNS, 32, 64))) {
            assertEquals(DType.Q4, shard.dType());
            assertEquals(4, shard.shape().first());
            assertEquals(32, shard.shape().last());
            assertQ4ShardEquals(full, shard, 0, 32);
        }
    }

    @Test
    public void rejectsUnalignedQ4ColumnShard() {
        writeQ4Matrix();

        try (DefaultWeightLoader loader = new DefaultWeightLoader(tempDir.toFile())) {
            assertThrows(IllegalArgumentException.class,
                    () -> loader.load("q4.weight", new TensorShardSpec(TensorShardAxis.COLUMNS, 16, 48)));
        }
    }

    @Test
    public void loadsGemmaLikeQ4MlpShardsAsLocalDenseTensors() {
        FloatBufferTensor gate = deterministicMatrix(128, 64, 0.03125f);
        FloatBufferTensor up = deterministicMatrix(128, 64, -0.0175f);
        FloatBufferTensor down = deterministicMatrix(64, 128, 0.0225f);
        SafeTensorWriter.write(tempDir.resolve("model.safetensors"), Map.of(), Map.of(
                "model.layers.0.mlp.gate_proj.weight", new Q4ByteBufferTensor(gate),
                "model.layers.0.mlp.up_proj.weight", new Q4ByteBufferTensor(up),
                "model.layers.0.mlp.down_proj.weight", new Q4ByteBufferTensor(down)));

        try (DefaultWeightLoader loader = new DefaultWeightLoader(tempDir.toFile());
             AbstractTensor fullGate = loader.load("model.layers.0.mlp.gate_proj.weight");
             AbstractTensor gateShard = loader.load("model.layers.0.mlp.gate_proj.weight",
                     new TensorShardSpec(TensorShardAxis.ROWS, 32, 64));
             AbstractTensor fullUp = loader.load("model.layers.0.mlp.up_proj.weight");
             AbstractTensor upShard = loader.load("model.layers.0.mlp.up_proj.weight",
                     new TensorShardSpec(TensorShardAxis.ROWS, 64, 96));
             AbstractTensor fullDown = loader.load("model.layers.0.mlp.down_proj.weight");
             AbstractTensor downShard = loader.load("model.layers.0.mlp.down_proj.weight",
                     new TensorShardSpec(TensorShardAxis.COLUMNS, 32, 64))) {
            assertEquals(DType.Q4, gateShard.dType());
            assertEquals(DType.Q4, upShard.dType());
            assertEquals(DType.Q4, downShard.dType());
            assertQ4ShardEquals(fullGate, gateShard, 32, 0);
            assertQ4ShardEquals(fullUp, upShard, 64, 0);
            assertQ4ShardEquals(fullDown, downShard, 0, 32);
        } finally {
            gate.close();
            up.close();
            down.close();
        }
    }

    @Test
    public void loadedQ4RowShardProducesSamePanamaProjectionAsFullTensor() {
        int batchSize = 13;
        int embeddingLength = 2304;
        int fullRows = 2048;
        int shardRows = 512;
        FloatBufferTensor input = deterministicMatrix(batchSize, embeddingLength, 1.0f / 64.0f);
        FloatBufferTensor weights = deterministicMatrix(fullRows, embeddingLength, 1.0f / 80.0f);
        SafeTensorWriter.write(tempDir.resolve("model.safetensors"), Map.of(), Map.of(
                "projection.weight", new Q4ByteBufferTensor(weights)));

        try (DefaultWeightLoader loader = new DefaultWeightLoader(tempDir.toFile());
             AbstractTensor fullWeight = loader.load("projection.weight");
             AbstractTensor shardWeight = loader.load("projection.weight",
                     new TensorShardSpec(TensorShardAxis.ROWS, 0, shardRows));
             AbstractTensor fullOutput = new FloatBufferTensor(batchSize, fullRows);
             AbstractTensor shardOutput = new FloatBufferTensor(batchSize, shardRows);
             WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores())) {
            PanamaTensorOperations ops = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE,
                    Mockito.mock(TensorAllocator.class), pool);
            ops.batchDotProduct(fullOutput, input, fullWeight, 0, 0, embeddingLength, 0, 0, fullRows);
            ops.batchDotProduct(shardOutput, input, shardWeight, 0, 0, embeddingLength, 0, 0, shardRows);
            assertProjectionShardEquals(fullOutput, shardOutput, batchSize - 1, 0.0001f);
        } finally {
            input.close();
            weights.close();
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

    private void writeQ4Matrix() {
        FloatBufferTensor matrix = new FloatBufferTensor(4, 64);
        int value = -100;
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 64; col++) {
                matrix.set(value++, row, col);
            }
        }
        SafeTensorWriter.write(tempDir.resolve("model.safetensors"), Map.of(), Map.of("q4.weight", new Q4ByteBufferTensor(matrix)));
    }

    private void writeWideQ4Matrix() {
        FloatBufferTensor matrix = wideMatrix();
        SafeTensorWriter.write(tempDir.resolve("model.safetensors"), Map.of(),
                Map.of("wide_q4.weight", new Q4ByteBufferTensor(matrix)));
    }

    private void writeWideQ4IndexedModel() {
        FloatBufferTensor matrix = wideMatrix();
        SafeTensorWriter.writeModel(tempDir, Map.of(), Map.of("wide_q4.weight", new Q4ByteBufferTensor(matrix)), 256 * 1024);
    }

    private static FloatBufferTensor wideMatrix() {
        FloatBufferTensor matrix = new FloatBufferTensor(128, 2304);
        for (int row = 0; row < matrix.shape().first(); row++) {
            for (int col = 0; col < matrix.shape().last(); col++) {
                matrix.set(((row * 31 + col * 17) % 257 - 128) / 128.0f, row, col);
            }
        }
        return matrix;
    }

    private static FloatBufferTensor deterministicMatrix(int rows, int cols, float scale) {
        FloatBufferTensor matrix = new FloatBufferTensor(rows, cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                matrix.set(((row * 31 + col * 17) % 257 - 128) * scale, row, col);
            }
        }
        return matrix;
    }

    private static void assertProjectionShardEquals(AbstractTensor full, AbstractTensor shard, int batchRow,
            float tolerance) {
        for (int col = 0; col < shard.shape().last(); col++) {
            float expected = full.get(batchRow, col);
            float actual = shard.get(batchRow, col);
            assertEquals(expected, actual, tolerance,
                    "batchRow=" + batchRow + " col=" + col + " expected=" + expected + " actual=" + actual);
        }
    }

    private static void assertQ4ShardEquals(AbstractTensor full, AbstractTensor shard, int rowOffset, int colOffset) {
        for (int row = 0; row < shard.shape().first(); row++) {
            for (int col = 0; col < shard.shape().last(); col++) {
                assertEquals(full.get(row + rowOffset, col + colOffset), shard.get(row, col), 0.0f,
                        "row=" + row + " col=" + col);
            }
        }
    }

    private static String normalize(String display) {
        return display.strip().replaceAll("(?m) +$", "");
    }
}
