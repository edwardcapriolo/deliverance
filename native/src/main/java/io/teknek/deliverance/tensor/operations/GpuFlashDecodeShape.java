package io.teknek.deliverance.tensor.operations;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.AbstractTensor;

final class GpuFlashDecodeShape {
    private GpuFlashDecodeShape() {
    }

    static void validateVllmLayout(AbstractTensor valueOut, AbstractTensor query, AbstractTensor keyCache,
            AbstractTensor valueCache, int[] blockTable, int visibleRows, int blockSize, int numberOfHeads,
            int numberOfKeyValueHeads, int headSize) {
        if (query.dType() != DType.F32 || keyCache.dType() != DType.F32 || valueCache.dType() != DType.F32
                || valueOut.dType() != DType.F32) {
            throw new IllegalArgumentException("vLLM-layout GPU debug path requires F32 query/K/V/out tensors");
        }
        if (headSize > 128) {
            throw new IllegalArgumentException("vLLM-layout GPU debug path currently supports headSize <= 128");
        }
        if (query.shape().last() < numberOfHeads * headSize || valueOut.shape().last() < numberOfHeads * headSize) {
            throw new IllegalArgumentException("query/valueOut width does not contain all heads");
        }
        int kvLength = numberOfKeyValueHeads * headSize;
        if (keyCache.shape().last() < kvLength || valueCache.shape().last() < kvLength) {
            throw new IllegalArgumentException("KV cache row width does not contain all KV heads");
        }
        int requiredBlocks = (visibleRows + blockSize - 1) / blockSize;
        if (blockTable.length < requiredBlocks) {
            throw new IllegalArgumentException("blockTable shorter than visible row block count");
        }
        int cacheRows = keyCache.shape().first();
        for (int i = 0; i < requiredBlocks; i++) {
            int block = blockTable[i];
            if (block < 0 || (long) (block + 1) * blockSize > cacheRows) {
                throw new IllegalArgumentException("blockTable contains physical block outside KV cache: " + block);
            }
        }
    }
}
