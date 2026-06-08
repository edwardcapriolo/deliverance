package io.teknek.deliverance.model.tensorparallel;

import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorDisplayUtil;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.NaiveTensorOperations;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class TensorParallelMlpTest {

    @Test
    public void summedRankPartialsMatchFullMlpOutput() {
        ConfigurableTensorProvider provider = new ConfigurableTensorProvider(new NaiveTensorOperations());
        try (AbstractTensor input = input();
             AbstractTensor gate = matrix(4, 3, 0.10f);
             AbstractTensor up = matrix(4, 3, 0.20f);
             AbstractTensor down = matrix(3, 4, -0.15f);
             AbstractTensor gateRank0 = rowShard(gate, 0, 2);
             AbstractTensor upRank0 = rowShard(up, 0, 2);
             AbstractTensor downRank0 = columnShard(down, 0, 2);
             AbstractTensor gateRank1 = rowShard(gate, 2, 4);
             AbstractTensor upRank1 = rowShard(up, 2, 4);
             AbstractTensor downRank1 = columnShard(down, 2, 4);
             AbstractTensor full = TensorParallelMlp.forwardPartial(input, gate, up, down,
                     ActivationFunction.Type.SILU, provider, FloatBufferTensor::new);
             AbstractTensor rank1Partial = TensorParallelMlp.forwardPartial(input, gateRank1, upRank1, downRank1,
                     ActivationFunction.Type.SILU, provider, FloatBufferTensor::new);
             AbstractTensor summed = TensorParallelMlp.forward(input, gateRank0, upRank0, downRank0,
                     ActivationFunction.Type.SILU, provider, FloatBufferTensor::new,
                     new FixedSumCollectives(rank1Partial), "layer.0.mlp.down_proj")) {

            String expected = """
                    [0][0]= -5.0217 [0][1]=-10.9343 [0][2]=-16.8470
                    """.trim();

            assertEquals(expected, normalize(TensorDisplayUtil.pretty2dDisplayAll(full)));
            assertEquals(expected, normalize(TensorDisplayUtil.pretty2dDisplayAll(summed)));
        }
    }

    private static AbstractTensor input() {
        AbstractTensor input = new FloatBufferTensor(1, 3);
        input.set(0.5f, 0, 0);
        input.set(-1.0f, 0, 1);
        input.set(2.0f, 0, 2);
        return input;
    }

    private static AbstractTensor matrix(int rows, int cols, float scale) {
        AbstractTensor tensor = new FloatBufferTensor(rows, cols);
        int value = 1;
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                tensor.set(value++ * scale, row, col);
            }
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
