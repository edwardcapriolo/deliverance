package io.teknek.deliverance.model.granitemoehybrid;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.TensorDisplayUtil;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.impl.Q4ByteBufferTensor;
import io.teknek.deliverance.tensor.operations.MachineSpec;
import io.teknek.deliverance.tensor.operations.PanamaTensorOperations;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class GraniteMoeHybridEmbeddingBatchCharacterizationTest {

    @Test
    void directBatchEmbeddingMatchesGenericPathWithLessWork() {
        int hidden = 64;
        int[] tokenIds = {0, 3, 5, 2};
        float scale = 2.0f;
        Q4ByteBufferTensor embeddingTable = q4EmbeddingTable(8, hidden);
        CountingOps ops = new CountingOps();

        long oldStart = System.nanoTime();
        AbstractTensor oldBatch = genericBatchEmbeddingPath(embeddingTable, tokenIds, hidden, scale, ops);
        long oldElapsedNanos = System.nanoTime() - oldStart;
        int oldSingleEmbeddingAllocations = ops.singleEmbeddingAllocations;
        int oldScaleCalls = ops.scaleCalls;

        ops.resetCounts();
        long directStart = System.nanoTime();
        AbstractTensor directBatch = directBatchEmbeddingPath(embeddingTable, tokenIds, hidden, scale, ops);
        long directElapsedNanos = System.nanoTime() - directStart;

        System.out.printf(java.util.Locale.ROOT,
                "Granite embedding batch characterization: old=%.3fms direct=%.3fms single_embedding_allocations=%d->%d scale_calls=%d->%d%n",
                oldElapsedNanos / 1_000_000.0,
                directElapsedNanos / 1_000_000.0,
                oldSingleEmbeddingAllocations,
                ops.singleEmbeddingAllocations,
                oldScaleCalls,
                ops.scaleCalls);

        assertEquals(oldBatch.shape(), directBatch.shape());
        assertEquals(TensorDisplayUtil.pretty2dDisplayAll(oldBatch).trim(),
                TensorDisplayUtil.pretty2dDisplayAll(directBatch).trim());
        assertEquals(tokenIds.length, oldSingleEmbeddingAllocations);
        assertEquals(0, ops.singleEmbeddingAllocations);
        assertEquals(tokenIds.length, oldScaleCalls);
        assertEquals(1, ops.scaleCalls);
    }

    private static AbstractTensor genericBatchEmbeddingPath(Q4ByteBufferTensor embeddingTable, int[] tokenIds,
            int hidden, float scale, CountingOps ops) {
        AbstractTensor first = singleTokenEmbedding(embeddingTable, tokenIds[0], hidden, scale, ops);
        AbstractTensor batch = ops.allocate(DType.F32, TensorShape.of(tokenIds.length, hidden));
        batch.copyFrom(first, 0, 0, hidden);
        first.close();
        for (int i = 1; i < tokenIds.length; i++) {
            AbstractTensor embedding = singleTokenEmbedding(embeddingTable, tokenIds[i], hidden, scale, ops);
            batch.copyFrom(embedding, 0, batch.getOffset(i, 0), hidden);
            embedding.close();
        }
        return batch;
    }

    private static AbstractTensor singleTokenEmbedding(Q4ByteBufferTensor embeddingTable, int tokenId, int hidden,
            float scale, CountingOps ops) {
        AbstractTensor row = embeddingTable.slice(true, tokenId);
        AbstractTensor converted = ops.panama.quantize(row, DType.F32, 0, hidden);
        AbstractTensor embedding = ops.allocateSingleEmbedding(TensorShape.of(1, hidden));
        embedding.copyFrom(converted, 0, 0, hidden);
        converted.close();
        ops.scale(scale, embedding, 0, hidden);
        return embedding;
    }

    private static AbstractTensor directBatchEmbeddingPath(Q4ByteBufferTensor embeddingTable, int[] tokenIds,
            int hidden, float scale, CountingOps ops) {
        AbstractTensor batch = ops.allocate(DType.F32, TensorShape.of(tokenIds.length, hidden));
        for (int i = 0; i < tokenIds.length; i++) {
            AbstractTensor row = embeddingTable.slice(true, tokenIds[i]);
            AbstractTensor converted = ops.panama.quantize(row, DType.F32, 0, hidden);
            batch.copyFrom(converted, 0, batch.getOffset(i, 0), hidden);
            converted.close();
        }
        ops.scale(scale, batch, 0, hidden);
        return batch;
    }

    private static Q4ByteBufferTensor q4EmbeddingTable(int rows, int hidden) {
        FloatBufferTensor dense = new FloatBufferTensor(rows, hidden);
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < hidden; column++) {
                dense.set((row + 1) * ((column % 11) - 5.0f), row, column);
            }
        }
        return new Q4ByteBufferTensor(dense);
    }

    private static final class CountingOps implements AutoCloseable {
        private final TensorAllocator allocator = new ArrayQueueTensorAllocator(new MetricRegistry());
        private final WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores());
        private final PanamaTensorOperations panama = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE, allocator, pool);
        private int singleEmbeddingAllocations;
        private int scaleCalls;

        private AbstractTensor allocate(DType dType, TensorShape shape) {
            return allocator.getDirty(dType, shape);
        }

        private AbstractTensor allocateSingleEmbedding(TensorShape shape) {
            singleEmbeddingAllocations++;
            return allocate(DType.F32, shape);
        }

        private void scale(float factor, AbstractTensor target, int offset, int length) {
            scaleCalls++;
            panama.scale(factor, target, offset, length);
        }

        private void resetCounts() {
            singleEmbeddingAllocations = 0;
            scaleCalls = 0;
        }

        @Override
        public void close() {
            pool.close();
        }
    }
}
