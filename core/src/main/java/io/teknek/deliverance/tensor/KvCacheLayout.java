package io.teknek.deliverance.tensor;

import io.teknek.deliverance.tensor.AbstractTensor;

/** Utility operations for vLLM-style KV slot/block mapping over flat row-major cache tensors. */
public final class KvCacheLayout {
    private KvCacheLayout() {
    }

    public static void reshapeAndCache(AbstractTensor key, AbstractTensor value, AbstractTensor keyCache,
            AbstractTensor valueCache, KvSlotMapping slotMapping, int rowWidth) {
        for (int token = 0; token < slotMapping.tokenCount(); token++) {
            int slot = slotMapping.slot(token);
            keyCache.copyFrom(key, key.getOffset(token, 0), keyCache.getOffset(slot, 0), rowWidth);
            valueCache.copyFrom(value, value.getOffset(token, 0), valueCache.getOffset(slot, 0), rowWidth);
        }
    }

    public static void gather(AbstractTensor sourceCache, AbstractTensor destination, KvBlockTable blockTable,
            int sequence, int startPosition, int length, int rowWidth) {
        for (int i = 0; i < length; i++) {
            int slot = blockTable.slot(sequence, startPosition + i);
            destination.copyFrom(sourceCache, sourceCache.getOffset(slot, 0), destination.getOffset(i, 0), rowWidth);
        }
    }
}
