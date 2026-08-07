package io.teknek.deliverance.tensor;

import com.google.common.base.Preconditions;

import java.util.Arrays;

/**
 * vLLM-style mapping from token index to a physical KV slot.
 *
 * <p>A slot is {@code physicalBlock * blockSize + blockOffset}. Deliverance currently uses logical token position as
 * the slot for normal single-sequence decode, but this explicit type lets tests and future GPU cache code exercise the
 * same slot-mapping semantics as vLLM.</p>
 */
public record KvSlotMapping(int blockSize, int[] slots) {

    public KvSlotMapping {
        Preconditions.checkArgument(blockSize > 0, "blockSize must be positive");
        slots = Arrays.copyOf(slots, slots.length);
        for (int slot : slots) {
            Preconditions.checkArgument(slot >= 0, "slot must be non-negative");
        }
    }

    public static KvSlotMapping contiguous(int blockSize, int tokenCount) {
        int[] slots = new int[tokenCount];
        for (int i = 0; i < tokenCount; i++) {
            slots[i] = i;
        }
        return new KvSlotMapping(blockSize, slots);
    }

    public int tokenCount() {
        return slots.length;
    }

    public int slot(int tokenIndex) {
        return slots[tokenIndex];
    }

    public int blockIndex(int tokenIndex) {
        return slot(tokenIndex) / blockSize;
    }

    public int blockOffset(int tokenIndex) {
        return slot(tokenIndex) % blockSize;
    }
}
