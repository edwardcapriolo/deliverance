package io.teknek.deliverance.tensor;

import com.google.common.base.Preconditions;

import java.util.Arrays;

/** vLLM-style logical-block to physical-block table. */
public record KvBlockTable(int blockSize, int[][] blocks) {

    public KvBlockTable {
        Preconditions.checkArgument(blockSize > 0, "blockSize must be positive");
        int[][] copy = new int[blocks.length][];
        for (int sequence = 0; sequence < blocks.length; sequence++) {
            copy[sequence] = Arrays.copyOf(blocks[sequence], blocks[sequence].length);
            for (int block : copy[sequence]) {
                Preconditions.checkArgument(block >= 0, "physical block must be non-negative");
            }
        }
        blocks = copy;
    }

    public int sequenceCount() {
        return blocks.length;
    }

    public int physicalBlock(int sequence, int logicalBlock) {
        return blocks[sequence][logicalBlock];
    }

    public int slot(int sequence, int logicalPosition) {
        int logicalBlock = logicalPosition / blockSize;
        int blockOffset = logicalPosition % blockSize;
        return physicalBlock(sequence, logicalBlock) * blockSize + blockOffset;
    }
}
