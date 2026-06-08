package io.teknek.deliverance.model.tensorparallel;

import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorDisplayUtil;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class TensorParallelAttentionTest {

    @Test
    public void summedRankPartialsMatchFullCausalAttentionOutput() {
        try (AbstractTensor input = input();
             AbstractTensor q = identity(4);
             AbstractTensor k = identity(4);
             AbstractTensor v = identity(4);
             AbstractTensor o = identity(4);
             AbstractTensor qRank0 = rowShard(q, 0, 2);
             AbstractTensor kRank0 = rowShard(k, 0, 2);
             AbstractTensor vRank0 = rowShard(v, 0, 2);
             AbstractTensor oRank0 = columnShard(o, 0, 2);
             AbstractTensor qRank1 = rowShard(q, 2, 4);
             AbstractTensor kRank1 = rowShard(k, 2, 4);
             AbstractTensor vRank1 = rowShard(v, 2, 4);
             AbstractTensor oRank1 = columnShard(o, 2, 4);
             AbstractTensor full = TensorParallelAttention.forwardPartial(input, q, k, v, o, 2, 2, 1.0f,
                     FloatBufferTensor::new);
             AbstractTensor rank1Partial = TensorParallelAttention.forwardPartial(input, qRank1, kRank1, vRank1,
                     oRank1, 1, 2, 1.0f, FloatBufferTensor::new);
             AbstractTensor summed = TensorParallelAttention.forward(input, qRank0, kRank0, vRank0, oRank0, 1, 2,
                     1.0f, FloatBufferTensor::new, new FixedSumCollectives(rank1Partial),
                     "layer.0.self_attn.o_proj")) {

            String expected = """
                    [0][0]=  1.0000 [0][1]=  0.0000 [0][2]=  0.0000 [0][3]=  1.0000
                    [1][0]=  0.2689 [1][1]=  0.7311 [1][2]=  0.7311 [1][3]=  0.2689
                    """.trim();

            assertEquals(expected, normalize(TensorDisplayUtil.pretty2dDisplayAll(full)));
            assertEquals(expected, normalize(TensorDisplayUtil.pretty2dDisplayAll(summed)));
        }
    }

    private static AbstractTensor input() {
        AbstractTensor input = new FloatBufferTensor(2, 4);
        input.set(1.0f, 0, 0);
        input.set(0.0f, 0, 1);
        input.set(0.0f, 0, 2);
        input.set(1.0f, 0, 3);
        input.set(0.0f, 1, 0);
        input.set(1.0f, 1, 1);
        input.set(1.0f, 1, 2);
        input.set(0.0f, 1, 3);
        return input;
    }

    private static AbstractTensor identity(int size) {
        AbstractTensor tensor = new FloatBufferTensor(size, size);
        for (int i = 0; i < size; i++) {
            tensor.set(1.0f, i, i);
        }
        return tensor;
    }

    private static AbstractTensor rowShard(AbstractTensor source, int startInclusive, int endExclusive) {
        int rows = endExclusive - startInclusive;
        int cols = source.shape().last();
        AbstractTensor shard = new FloatBufferTensor(rows, cols);
        for (int row = 0; row < rows; row++) {
            shard.copyFrom(source, source.getOffset(startInclusive + row, 0), shard.getOffset(row, 0), cols);
        }
        return shard;
    }

    private static AbstractTensor columnShard(AbstractTensor source, int startInclusive, int endExclusive) {
        int rows = source.shape().first();
        int cols = endExclusive - startInclusive;
        AbstractTensor shard = new FloatBufferTensor(rows, cols);
        for (int row = 0; row < rows; row++) {
            shard.copyFrom(source, source.getOffset(row, startInclusive), shard.getOffset(row, 0), cols);
        }
        return shard;
    }

    private static AbstractTensor sum(AbstractTensor first, AbstractTensor second) {
        AbstractTensor result = new FloatBufferTensor(first.shape());
        result.copyFrom(first, 0, 0, (int) first.size());
        for (int row = 0; row < result.shape().first(); row++) {
            for (int col = 0; col < result.shape().last(); col++) {
                result.set(result.get(row, col) + second.get(row, col), row, col);
            }
        }
        return result;
    }

    private static final class FixedSumCollectives implements TensorParallelCollectives {
        private final AbstractTensor otherPartial;

        private FixedSumCollectives(AbstractTensor otherPartial) {
            this.otherPartial = otherPartial;
        }

        @Override
        public AbstractTensor allReduceSum(String key, AbstractTensor local) {
            return sum(local, otherPartial);
        }
    }

    private static String normalize(String display) {
        return display.strip().replaceAll("(?m) +$", "");
    }
}
