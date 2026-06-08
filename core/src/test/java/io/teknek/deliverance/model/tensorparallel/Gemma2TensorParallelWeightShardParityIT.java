package io.teknek.deliverance.model.tensorparallel;

import io.teknek.deliverance.safetensors.DefaultWeightLoader;
import io.teknek.deliverance.safetensors.TensorShardAxis;
import io.teknek.deliverance.safetensors.TensorShardSpec;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.impl.Q4ByteBufferTensor;
import org.junit.jupiter.api.Test;

import java.lang.foreign.ValueLayout;
import java.io.File;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class Gemma2TensorParallelWeightShardParityIT {
    private static final int TP_SIZE = 4;

    @Test
    public void layerZeroTensorParallelShardsMatchFullWeights() {
        File modelRoot = new ModelFetcher("tjake", "gemma-2-2b-it-JQ4").maybeDownload();
        try (DefaultWeightLoader loader = new DefaultWeightLoader(modelRoot)) {
            assertRowShardsMatch(loader, "model.layers.0.self_attn.q_proj.weight");
            assertRowShardsMatch(loader, "model.layers.0.self_attn.k_proj.weight");
            assertRowShardsMatch(loader, "model.layers.0.self_attn.v_proj.weight");
            assertColumnShardsMatch(loader, "model.layers.0.self_attn.o_proj.weight");
            assertRowShardsMatch(loader, "model.layers.0.mlp.gate_proj.weight");
            assertDirectRowShardsMatch(loader, "model.layers.0.mlp.gate_proj.weight.qb");
            assertRowShardsMatch(loader, "model.layers.0.mlp.up_proj.weight");
            assertDirectRowShardsMatch(loader, "model.layers.0.mlp.up_proj.weight.qb");
            assertColumnShardsMatch(loader, "model.layers.0.mlp.down_proj.weight");
            assertDirectColumnShardsMatch(loader, "model.layers.0.mlp.down_proj.weight.qb");
        }
    }

    private static void assertRowShardsMatch(DefaultWeightLoader loader, String weightName) {
        try (AbstractTensor full = loader.load(weightName)) {
            for (int rank = 0; rank < TP_SIZE; rank++) {
                TensorParallelWeightLoader tpLoader = tensorParallelWeightLoader(loader, rank, full, weightName);
                ShardRange range = rowRangeFor(weightName, full, rank);
                try (AbstractTensor shard = tpLoader.load(weightName)) {
                    assertEquals(range.length(), shard.shape().first(), weightName + " rank=" + rank + " row count");
                    assertEquals(full.shape().last(), shard.shape().last(), weightName + " rank=" + rank + " column count");
                    assertShardValues(weightName, full, shard, rank, range.startInclusive(), 0, 0.0f);
                }
            }
        }
    }

    private static void assertColumnShardsMatch(DefaultWeightLoader loader, String weightName) {
        try (AbstractTensor full = loader.load(weightName)) {
            for (int rank = 0; rank < TP_SIZE; rank++) {
                TensorParallelWeightLoader tpLoader = tensorParallelWeightLoader(loader, rank, full, weightName);
                ShardRange range = columnRangeFor(weightName, full, rank);
                try (AbstractTensor shard = tpLoader.load(weightName)) {
                    assertEquals(full.shape().first(), shard.shape().first(), weightName + " rank=" + rank + " row count");
                    assertEquals(range.length(), shard.shape().last(), weightName + " rank=" + rank + " column count");
                    assertShardValues(weightName, full, shard, rank, 0, range.startInclusive(), 0.0f);
                }
            }
        }
    }

    private static void assertDirectRowShardsMatch(DefaultWeightLoader loader, String weightName) {
        try (AbstractTensor full = loader.load(weightName)) {
            for (int rank = 0; rank < TP_SIZE; rank++) {
                ShardRange range = rowRangeFor(weightName, full, rank);
                try (AbstractTensor shard = loader.load(weightName,
                        new TensorShardSpec(TensorShardAxis.ROWS, range.startInclusive(), range.endExclusive()))) {
                    assertEquals(range.length(), shard.shape().first(), weightName + " rank=" + rank + " row count");
                    assertEquals(full.shape().last(), shard.shape().last(), weightName + " rank=" + rank + " column count");
                    assertShardValues(weightName, full, shard, rank, range.startInclusive(), 0, 0.0f);
                }
            }
        }
    }

    private static void assertDirectColumnShardsMatch(DefaultWeightLoader loader, String weightName) {
        try (AbstractTensor full = loader.load(weightName)) {
            for (int rank = 0; rank < TP_SIZE; rank++) {
                ShardRange range = columnRangeFor(weightName, full, rank);
                try (AbstractTensor shard = loader.load(weightName,
                        new TensorShardSpec(TensorShardAxis.COLUMNS, range.startInclusive(), range.endExclusive()))) {
                    assertEquals(full.shape().first(), shard.shape().first(), weightName + " rank=" + rank + " row count");
                    assertEquals(range.length(), shard.shape().last(), weightName + " rank=" + rank + " column count");
                    assertShardValues(weightName, full, shard, rank, 0, range.startInclusive(), 0.0f);
                }
            }
        }
    }

    private static TensorParallelWeightLoader tensorParallelWeightLoader(DefaultWeightLoader loader, int rank,
            AbstractTensor full, String weightName) {
        TensorParallelContext context = new StaticTensorParallelContext(rank, TP_SIZE);
        TensorParallelShardPlan plan = planFor(weightName, full, context);
        return new TensorParallelWeightLoader(loader, context, plan, new DefaultTransformerWeightPolicyResolver());
    }

    private static TensorParallelShardPlan planFor(String weightName, AbstractTensor full, TensorParallelContext context) {
        ShardRange range = weightName.endsWith(".o_proj.weight") || weightName.endsWith(".down_proj.weight")
                ? columnRangeFor(weightName, full, context.rank())
                : rowRangeFor(weightName, full, context.rank());
        ShardRange unused = new ShardRange(0, 0);
        if (weightName.contains("self_attn.q_proj") || weightName.contains("self_attn.o_proj")) {
            return new TensorParallelShardPlan(unused, unused, range, unused, unused);
        }
        if (weightName.contains("self_attn.k_proj") || weightName.contains("self_attn.v_proj")) {
            return new TensorParallelShardPlan(unused, unused, unused, range, unused);
        }
        return new TensorParallelShardPlan(unused, unused, unused, unused, range);
    }

    private static ShardRange rowRangeFor(String weightName, AbstractTensor full, int rank) {
        if (full.shape().first() % TP_SIZE != 0) {
            throw new IllegalArgumentException(weightName + " rows are not divisible by " + TP_SIZE);
        }
        int shardRows = full.shape().first() / TP_SIZE;
        return new ShardRange(rank * shardRows, (rank + 1) * shardRows);
    }

    private static ShardRange columnRangeFor(String weightName, AbstractTensor full, int rank) {
        if (full.shape().last() % TP_SIZE != 0) {
            throw new IllegalArgumentException(weightName + " columns are not divisible by " + TP_SIZE);
        }
        int shardColumns = full.shape().last() / TP_SIZE;
        return new ShardRange(rank * shardColumns, (rank + 1) * shardColumns);
    }

    private static void assertShardValues(String weightName, AbstractTensor full, AbstractTensor shard, int rank,
            int rowOffset, int columnOffset, float tolerance) {
        for (int row = 0; row < shard.shape().first(); row++) {
            for (int col = 0; col < shard.shape().last(); col++) {
                float expected = full.get(row + rowOffset, col + columnOffset);
                float actual = shard.get(row, col);
                int localRow = row;
                int localCol = col;
                assertEquals(expected, actual, tolerance, () -> mismatchMessage(weightName, full, shard, rank,
                        localRow, localCol, rowOffset, columnOffset, expected, actual));
            }
        }
    }

    private static String mismatchMessage(String weightName, AbstractTensor full, AbstractTensor shard, int rank,
            int localRow, int localCol, int rowOffset, int columnOffset, float expected, float actual) {
        int fullRow = localRow + rowOffset;
        int fullCol = localCol + columnOffset;
        String message = weightName + " rank=" + rank
                + " fullShape=" + full.shape()
                + " shardShape=" + shard.shape()
                + " fullDType=" + full.dType()
                + " shardDType=" + shard.dType()
                + " rowOffset=" + rowOffset
                + " columnOffset=" + columnOffset
                + " local=(" + localRow + "," + localCol + ") full=(" + fullRow + "," + fullCol + ")"
                + " expected=" + expected + " actual=" + actual
                + q4BlockScaleMessage(full, shard, fullRow, fullCol, localRow, localCol);
        return message;
    }

    private static String q4BlockScaleMessage(AbstractTensor full, AbstractTensor shard, int fullRow, int fullCol,
            int localRow, int localCol) {
        if (!(full instanceof Q4ByteBufferTensor fullQ4) || !(shard instanceof Q4ByteBufferTensor shardQ4)) {
            return "";
        }
        int fullBlockColumn = fullCol / Q4ByteBufferTensor.BLOCK_SIZE;
        int localBlockColumn = localCol / Q4ByteBufferTensor.BLOCK_SIZE;
        float fullScale = fullQ4.getBlockF().get(fullRow, fullBlockColumn);
        float shardScale = shardQ4.getBlockF().get(localRow, localBlockColumn);
        int fullLogicalOffset = fullQ4.getOffset(fullRow, fullCol);
        int shardLogicalOffset = shardQ4.getOffset(localRow, localCol);
        int fullByteOffset = q4PackedByteOffset(fullLogicalOffset);
        int shardByteOffset = q4PackedByteOffset(shardLogicalOffset);
        byte fullByte = fullQ4.getMemorySegment().get(ValueLayout.JAVA_BYTE, fullByteOffset);
        byte shardByte = shardQ4.getMemorySegment().get(ValueLayout.JAVA_BYTE, shardByteOffset);
        return " fullQ4Scale=(" + fullRow + "," + fullBlockColumn + ")=" + fullScale
                + " shardQ4Scale=(" + localRow + "," + localBlockColumn + ")=" + shardScale
                + " fullQ4ByteOffset=" + fullByteOffset
                + " shardQ4ByteOffset=" + shardByteOffset
                + " fullQ4Byte=" + Byte.toUnsignedInt(fullByte)
                + " shardQ4Byte=" + Byte.toUnsignedInt(shardByte)
                + " fullQ4Nibble=" + q4Nibble(fullByte, fullLogicalOffset)
                + " shardQ4Nibble=" + q4Nibble(shardByte, shardLogicalOffset);
    }

    private static int q4PackedByteOffset(int logicalOffset) {
        int offsetInBlock = logicalOffset % Q4ByteBufferTensor.BLOCK_SIZE;
        int byteOffset = (logicalOffset / Q4ByteBufferTensor.BLOCK_SIZE) * Q4ByteBufferTensor.HALF_BLOCK
                + offsetInBlock;
        return offsetInBlock < Q4ByteBufferTensor.HALF_BLOCK
                ? byteOffset
                : byteOffset - Q4ByteBufferTensor.HALF_BLOCK;
    }

    private static int q4Nibble(byte packedByte, int logicalOffset) {
        int unsigned = Byte.toUnsignedInt(packedByte);
        int offsetInBlock = logicalOffset % Q4ByteBufferTensor.BLOCK_SIZE;
        return offsetInBlock < Q4ByteBufferTensor.HALF_BLOCK
                ? (unsigned & 0x0F) - 8
                : ((unsigned >> 4) & 0x0F) - 8;
    }
}
