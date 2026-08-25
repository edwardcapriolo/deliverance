package io.teknek.deliverance.tensor.operations;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.tensor.AbstractTensor;

public final class ParallelSplitSizedTensorOperations implements TensorOperations {
    private final TensorOperations delegate;
    private final int parallelSplitSize;

    public ParallelSplitSizedTensorOperations(TensorOperations delegate, int parallelSplitSize) {
        if (parallelSplitSize < 1) {
            throw new IllegalArgumentException("parallelSplitSize must be >= 1");
        }
        this.delegate = java.util.Objects.requireNonNull(delegate, "delegate");
        this.parallelSplitSize = parallelSplitSize;
    }

    @Override
    public String name() {
        return delegate.name();
    }

    @Override
    public int parallelSplitSize() {
        return parallelSplitSize;
    }

    @Override
    public DType preferredWorkingQuantizedType() {
        return delegate.preferredWorkingQuantizedType();
    }

    @Override
    public void registerModelTensor(AbstractTensor t) {
        delegate.registerModelTensor(t);
    }

    @Override
    public float dotProduct(AbstractTensor a, AbstractTensor b, int aoffset, int boffset, int limit) {
        return delegate.dotProduct(a, b, aoffset, boffset, limit);
    }

    @Override
    public void batchDotProduct(AbstractTensor result, AbstractTensor a, AbstractTensor b, int aColumnOffset,
            int bColumnOffset, int columnLimit, int rRowOffset, int bRowOffset, int rowChunkSize) {
        delegate.batchDotProduct(result, a, b, aColumnOffset, bColumnOffset, columnLimit, rRowOffset, bRowOffset,
                rowChunkSize);
    }

    @Override
    public void dotProductBatchChunk(AbstractTensor[] result, AbstractTensor a, AbstractTensor[] b, int offset,
            int limit, int chunkStart, int chunkSize) {
        delegate.dotProductBatchChunk(result, a, b, offset, limit, chunkStart, chunkSize);
    }

    @Override
    public void accumulate(AbstractTensor a, AbstractTensor b, int offset, int length) {
        delegate.accumulate(a, b, offset, length);
    }

    @Override
    public void maccumulate(AbstractTensor a, AbstractTensor b, int offset, int length) {
        delegate.maccumulate(a, b, offset, length);
    }

    @Override
    public void saxpy(float alpha, AbstractTensor x, AbstractTensor y, int xoffset, int yoffset, int limit) {
        delegate.saxpy(alpha, x, y, xoffset, yoffset, limit);
    }

    @Override
    public void saxpy(AbstractTensor alpha, AbstractTensor x, AbstractTensor y, int xoffset, int yoffset, int limit,
            int aOffset, int xRowOffset, int batchSize) {
        delegate.saxpy(alpha, x, y, xoffset, yoffset, limit, aOffset, xRowOffset, batchSize);
    }

    @Override
    public void scale(float factor, AbstractTensor x, int offset, int length) {
        delegate.scale(factor, x, offset, length);
    }

    @Override
    public AbstractTensor quantize(AbstractTensor t, DType qtype, int offset, int length) {
        return delegate.quantize(t, qtype, offset, length);
    }

    @Override
    public AbstractTensor activationMultiplyQuantize(AbstractTensor gate, AbstractTensor up,
            ActivationFunction.Type activation, DType qtype, int offset, int length) {
        return delegate.activationMultiplyQuantize(gate, up, activation, qtype, offset, length);
    }

    @Override
    public void gatherRowsAdd(AbstractTensor output, AbstractTensor wordEmbeddings, int[] inputIds,
            AbstractTensor tokenTypeEmbeddings, int[] tokenTypeIds, AbstractTensor positionEmbeddings,
            int[] positionIds, int rowOffset, int rowCount) {
        delegate.gatherRowsAdd(output, wordEmbeddings, inputIds, tokenTypeEmbeddings, tokenTypeIds,
                positionEmbeddings, positionIds, rowOffset, rowCount);
    }

    @Override
    public void decodePagedAttention(AbstractTensor valueOut, AbstractTensor query, AbstractTensor[] keyPages,
            AbstractTensor[] valuePages, int visibleRows, int numberOfHeads, int numberOfKeyValueHeads, int headSize,
            float scale, Float softcap) {
        delegate.decodePagedAttention(valueOut, query, keyPages, valuePages, visibleRows, numberOfHeads,
                numberOfKeyValueHeads, headSize, scale, softcap);
    }

    @Override
    public boolean supportsDecodePagedAttention(AbstractTensor valueOut, AbstractTensor query, AbstractTensor[] keyPages,
            AbstractTensor[] valuePages, int visibleRows, int numberOfHeads, int numberOfKeyValueHeads, int headSize,
            float scale, Float softcap) {
        return delegate.supportsDecodePagedAttention(valueOut, query, keyPages, valuePages, visibleRows,
                numberOfHeads, numberOfKeyValueHeads, headSize, scale, softcap);
    }

    @Override
    public void decodePagedAttentionHeadWithProviderKernels(AbstractTensor valueOut, AbstractTensor query,
            AbstractTensor[] keyPages, AbstractTensor[] valuePages, int visibleRows, int numberOfHeads,
            int numberOfKeyValueHeads, int headSize, float scale, Float softcap, int head) {
        delegate.decodePagedAttentionHeadWithProviderKernels(valueOut, query, keyPages, valuePages, visibleRows,
                numberOfHeads, numberOfKeyValueHeads, headSize, scale, softcap, head);
    }
}
