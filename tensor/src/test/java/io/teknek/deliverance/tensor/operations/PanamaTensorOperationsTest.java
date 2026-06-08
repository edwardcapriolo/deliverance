package io.teknek.deliverance.tensor.operations;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.tensor.*;
import io.teknek.deliverance.tensor.impl.BFloat16BufferTensor;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.impl.Q4ByteBufferTensor;
import io.teknek.deliverance.tensor.impl.Q8ByteBufferTensor;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class PanamaTensorOperationsTest {

    public static AbstractTensor allOnes(int size) {
        AbstractTensor f = new FloatBufferTensor(1, size);
        for (int i = 0; i < size; i++) {
            f.set(1.0f, 0, i);
        }
        return f;
    }

    public static AbstractTensor allZeros(int size) {
        AbstractTensor f = new FloatBufferTensor(1, size);
        for (int i = 0; i < size; i++) {
            f.set(0.0f, 0, i);
        }
        return f;
    }

    static AbstractTensor random(int size, int seed) {
        Random r = new Random(seed);
        AbstractTensor f = new FloatBufferTensor(1, size);
        for (int i = 0; i < size; i++) {
            f.set(r.nextFloat(-1, 100), 0, i);
        }
        return f;
    }

    @Test
    void simpleDotProduct() {
        int size = 1024;
        NaiveTensorOperations controlOps = new NaiveTensorOperations();
        AbstractTensor a = allOnes(size);
        AbstractTensor b = allOnes(size);
        float control = controlOps.dotProduct(a, b, size);
        assertEquals(1024f, control);
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores())) {
            PanamaTensorOperations p = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE, Mockito.mock(TensorAllocator.class), pool);
            assertEquals(control, p.dotProduct(a, b, 1024));
        }
    }


    @Test
    void dotProductQuantizationTest() {
        int size = 1024;
        int seed = 43;
        NaiveTensorOperations controlOps = new NaiveTensorOperations();
        AbstractTensor a = random(size, seed);
        AbstractTensor b = random(size, seed + 1);
        AbstractTensor q8 = new Q8ByteBufferTensor(a);
        AbstractTensor q4 = new Q4ByteBufferTensor(b);

        float expected = 2587953.2f;
        float control = controlOps.dotProduct(q8, q4, size);
        assertEquals(expected, control);
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores())) {
            PanamaTensorOperations p = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE, Mockito.mock(TensorAllocator.class), pool);
            assertEquals(control, p.dotProduct(q8, q4, size), control * .01f);
        }
    }

    @Test
    void dotProductChunkBf16Q4Test() {
        AbstractTensor a = new FloatBufferTensor(1, 32);
        for (int i = 0; i < 32; i++) {
            a.set(1.0f, 0, i);
        }
        BFloat16BufferTensor aBf16 = new BFloat16BufferTensor(a);
        Q4ByteBufferTensor q4 = (Q4ByteBufferTensor) AbstractTensorUtils.quantize(a, DType.Q4, true);
        FloatBufferTensor result = (FloatBufferTensor) PanamaTensorOperationsTest.allZeros(1);
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores())) {
            PanamaTensorOperations ops = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE, Mockito.mock(TensorAllocator.class), pool);
            ops.dotProductChunk(result, aBf16, q4, 0, 32, 0, 1);
        }
        assertEquals("[0][0]= 32.0000".trim(), TensorDisplayUtil.pretty2dDisplayAll(result).trim());
    }

    @Test
    void dotProductChunkBf16Q4AlternatingSigns() {
        AbstractTensor a = new FloatBufferTensor(1, 32);
        AbstractTensor b = new FloatBufferTensor(1, 32);
        for (int i = 0; i < 32; i++) {
            float av = (i % 2 == 0) ? 2.0f : -2.0f;
            float bv = (i % 2 == 0) ? 3.0f : -1.0f;
            a.set(av, 0, i);
            b.set(bv, 0, i);
        }
        BFloat16BufferTensor aBf16 = new BFloat16BufferTensor(a);
        Q4ByteBufferTensor q4 = (Q4ByteBufferTensor) AbstractTensorUtils.quantize(b, DType.Q4, true);
        FloatBufferTensor result = (FloatBufferTensor) PanamaTensorOperationsTest.allZeros(1);
        FloatBufferTensor expected = (FloatBufferTensor) PanamaTensorOperationsTest.allZeros(1);
        new NaiveTensorOperations().dotProductChunk(expected, aBf16, q4, 0, 32, 0, 1);
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores())) {
            PanamaTensorOperations ops = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE, Mockito.mock(TensorAllocator.class), pool);
            ops.dotProductChunk(result, aBf16, q4, 0, 32, 0, 1);
        }
        assertEquals(TensorDisplayUtil.pretty2dDisplayAll(expected).trim(), TensorDisplayUtil.pretty2dDisplayAll(result).trim());
    }

    @Test
    void dotProductChunkBf16Q4RampValues() {
        AbstractTensor a = new FloatBufferTensor(1, 32);
        AbstractTensor b = new FloatBufferTensor(1, 32);
        for (int i = 0; i < 32; i++) {
            a.set(i + 1, 0, i);
            b.set(2.0f, 0, i);
        }
        BFloat16BufferTensor aBf16 = new BFloat16BufferTensor(a);
        Q4ByteBufferTensor q4 = (Q4ByteBufferTensor) AbstractTensorUtils.quantize(b, DType.Q4, true);
        FloatBufferTensor result = (FloatBufferTensor) PanamaTensorOperationsTest.allZeros(1);
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores())) {
            PanamaTensorOperations ops = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE, Mockito.mock(TensorAllocator.class), pool);
            ops.dotProductChunk(result, aBf16, q4, 0, 32, 0, 1);
        }
        assertEquals("[0][0]=1056.0000".trim(), TensorDisplayUtil.pretty2dDisplayAll(result).trim());
    }

    @Test
    void dotProductChunkBf16Q4TwoOutputRows() {
        AbstractTensor a = new FloatBufferTensor(2, 32);
        AbstractTensor b = new FloatBufferTensor(2, 32);
        for (int i = 0; i < 32; i++) {
            a.set(1.0f, 0, i);
            a.set(2.0f, 1, i);
            b.set(3.0f, 0, i);
            b.set(4.0f, 1, i);
        }
        BFloat16BufferTensor aBf16 = new BFloat16BufferTensor(a);
        Q4ByteBufferTensor q4 = (Q4ByteBufferTensor) AbstractTensorUtils.quantize(b, DType.Q4, true);
        FloatBufferTensor result = new FloatBufferTensor(2, 2);
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores())) {
            PanamaTensorOperations ops = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE, Mockito.mock(TensorAllocator.class), pool);
            ops.dotProductChunk(result, aBf16, q4, 0, 32, 0, 2);
        }
        assertEquals("[0][0]= 96.0000 [0][1]=128.0000 \n[1][0]=192.0000 [1][1]=256.0000".trim(), TensorDisplayUtil.pretty2dDisplayAll(result).trim());
    }

    @Test
    void batchDotProductBf16Q4MultiRowMultiCol() {
        FloatBufferTensor a = new FloatBufferTensor(3, 256);
        FloatBufferTensor b = new FloatBufferTensor(4, 256);
        for (int row = 0; row < 3; row++) {
            for (int col = 0; col < 256; col++) {
                a.set((col % 7) - 3 + row, row, col);
            }
        }
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 256; col++) {
                b.set((col % 5) - 2 + row, row, col);
            }
        }
        BFloat16BufferTensor aBf16 = new BFloat16BufferTensor(a);
        Q4ByteBufferTensor q4 = (Q4ByteBufferTensor) AbstractTensorUtils.quantize(b, DType.Q4, true);
        FloatBufferTensor expected = new FloatBufferTensor(3, 4);
        FloatBufferTensor actual = new FloatBufferTensor(3, 4);

        new NaiveTensorOperations().batchDotProduct(expected, aBf16, q4, 0, 0, 256, 0, 0, 4);
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores())) {
            PanamaTensorOperations ops = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE, Mockito.mock(TensorAllocator.class), pool);
            ops.batchDotProduct(actual, aBf16, q4, 0, 0, 256, 0, 0, 4);
        }
        assertEquals(TensorDisplayUtil.pretty2dDisplayAll(expected).trim(), TensorDisplayUtil.pretty2dDisplayAll(actual).trim());
    }

    @Test
    void batchDotProductBf16Q4WithOffsetsAndRowChunk() {
        FloatBufferTensor a = new FloatBufferTensor(2, 256);
        FloatBufferTensor b = new FloatBufferTensor(5, 256);
        for (int row = 0; row < 2; row++) {
            for (int col = 0; col < 256; col++) {
                a.set((col % 9) - 4 + row, row, col);
            }
        }
        for (int row = 0; row < 5; row++) {
            for (int col = 0; col < 256; col++) {
                b.set((col % 6) - 3 + row, row, col);
            }
        }
        BFloat16BufferTensor aBf16 = new BFloat16BufferTensor(a);
        Q4ByteBufferTensor q4 = (Q4ByteBufferTensor) AbstractTensorUtils.quantize(b, DType.Q4, true);
        FloatBufferTensor expected = new FloatBufferTensor(2, 4);
        FloatBufferTensor actual = new FloatBufferTensor(2, 4);

        new NaiveTensorOperations().batchDotProduct(expected, aBf16, q4, 32, 32, 128, 0, 1, 3);
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores())) {
            PanamaTensorOperations ops = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE, Mockito.mock(TensorAllocator.class), pool);
            ops.batchDotProduct(actual, aBf16, q4, 32, 32, 128, 0, 1, 3);
        }
        assertEquals(TensorDisplayUtil.pretty2dDisplayAll(expected).trim(), TensorDisplayUtil.pretty2dDisplayAll(actual).trim());
    }

    @Test
    void batchDotProductF32Q4RowShardMatchesFullProjectionForRemainderBatchRow() {
        int batchSize = 13;
        int embeddingLength = 2304;
        int fullRows = 2048;
        int shardRows = 512;
        FloatBufferTensor input = deterministicInput(batchSize, embeddingLength);
        FloatBufferTensor fullWeightSource = deterministicWeight(fullRows, embeddingLength);
        FloatBufferTensor shardWeightSource = rowShard(fullWeightSource, 0, shardRows);
        Q4ByteBufferTensor fullWeight = new Q4ByteBufferTensor(fullWeightSource);
        Q4ByteBufferTensor shardWeight = new Q4ByteBufferTensor(shardWeightSource);
        FloatBufferTensor fullOutput = new FloatBufferTensor(batchSize, fullRows);
        FloatBufferTensor shardOutput = new FloatBufferTensor(batchSize, shardRows);

        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores())) {
            PanamaTensorOperations ops = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE,
                    Mockito.mock(TensorAllocator.class), pool);
            ops.batchDotProduct(fullOutput, input, fullWeight, 0, 0, embeddingLength, 0, 0, fullRows);
            ops.batchDotProduct(shardOutput, input, shardWeight, 0, 0, embeddingLength, 0, 0, shardRows);
        }

        assertRowShardProjectionEquals(fullOutput, shardOutput, batchSize - 1, 0.0001f);
    }

    private static FloatBufferTensor deterministicInput(int rows, int cols) {
        FloatBufferTensor tensor = new FloatBufferTensor(rows, cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                tensor.set(((row * 17 + col * 31) % 257 - 128) / 64.0f, row, col);
            }
        }
        return tensor;
    }

    private static FloatBufferTensor deterministicWeight(int rows, int cols) {
        FloatBufferTensor tensor = new FloatBufferTensor(rows, cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                tensor.set(((row * 43 + col * 19) % 251 - 125) / 80.0f, row, col);
            }
        }
        return tensor;
    }

    private static FloatBufferTensor rowShard(AbstractTensor source, int startInclusive, int length) {
        FloatBufferTensor shard = new FloatBufferTensor(length, source.shape().last());
        for (int row = 0; row < length; row++) {
            shard.copyFrom(source, source.getOffset(startInclusive + row, 0), shard.getOffset(row, 0),
                    source.shape().last());
        }
        return shard;
    }

    private static void assertRowShardProjectionEquals(AbstractTensor full, AbstractTensor shard, int batchRow,
            float tolerance) {
        for (int col = 0; col < shard.shape().last(); col++) {
            float expected = full.get(batchRow, col);
            float actual = shard.get(batchRow, col);
            assertEquals(expected, actual, tolerance,
                    "batchRow=" + batchRow + " col=" + col + " expected=" + expected + " actual=" + actual);
        }
    }
}
