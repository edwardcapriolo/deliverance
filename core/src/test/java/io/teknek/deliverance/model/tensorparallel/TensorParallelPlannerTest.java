package io.teknek.deliverance.model.tensorparallel;

import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.safetensors.Config;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class TensorParallelPlannerTest {

    @Test
    public void staticContextValidatesRankAndSize() {
        StaticTensorParallelContext context = new StaticTensorParallelContext(3, 4);

        assertEquals(3, context.rank());
        assertEquals(4, context.size());
        assertTrue(context.enabled());
        assertFalse(context.coordinatorRank());

        assertThrows(IllegalArgumentException.class, () -> new StaticTensorParallelContext(-1, 4));
        assertThrows(IllegalArgumentException.class, () -> new StaticTensorParallelContext(4, 4));
        assertThrows(IllegalArgumentException.class, () -> new StaticTensorParallelContext(0, 0));
    }

    @Test
    public void fourWayGemma2LikeConfigAssignsExpectedRangesForEachRank() {
        Config config = gemma2LikeConfig();

        assertTrue(TensorParallelPlanner.compatible(config, 4));

        assertPlan(0, 0, 2, 0, 1, 0, 512, 0, 256, 0, 2304);
        assertPlan(1, 2, 4, 1, 2, 512, 1024, 256, 512, 2304, 4608);
        assertPlan(2, 4, 6, 2, 3, 1024, 1536, 512, 768, 4608, 6912);
        assertPlan(3, 6, 8, 3, 4, 1536, 2048, 768, 1024, 6912, 9216);
    }

    @Test
    public void chooseSizeRoundsDownToLargestCompatibleNodeCount() {
        Config config = gemma2LikeConfig();

        assertEquals(4, TensorParallelPlanner.chooseSize(config, 10));
        assertEquals(4, TensorParallelPlanner.chooseSize(config, 4));
        assertEquals(2, TensorParallelPlanner.chooseSize(config, 3));
        assertEquals(1, TensorParallelPlanner.chooseSize(config, 1));
    }

    @Test
    public void rejectsIncompatibleShardSizes() {
        Config config = gemma2LikeConfig();

        assertFalse(TensorParallelPlanner.compatible(config, 3));
        assertThrows(IllegalArgumentException.class,
                () -> TensorParallelPlanner.plan(config, new StaticTensorParallelContext(0, 3)));
        assertThrows(IllegalArgumentException.class,
                () -> TensorParallelPlanner.range(10, new StaticTensorParallelContext(0, 4)));
    }

    @Test
    public void rankZeroIsCoordinatorForStaticContext() {
        StaticTensorParallelContext context = new StaticTensorParallelContext(0, 4);

        assertTrue(context.coordinatorRank());
    }

    private static void assertPlan(int rank,
            int qStart, int qEnd,
            int kvHeadStart, int kvHeadEnd,
            int attnStart, int attnEnd,
            int kvColumnStart, int kvColumnEnd,
            int mlpStart, int mlpEnd) {
        TensorParallelShardPlan plan = TensorParallelPlanner.plan(gemma2LikeConfig(), new StaticTensorParallelContext(rank, 4));

        assertEquals(new ShardRange(qStart, qEnd), plan.queryHeads());
        assertEquals(new ShardRange(kvHeadStart, kvHeadEnd), plan.keyValueHeads());
        assertEquals(new ShardRange(attnStart, attnEnd), plan.attentionColumns());
        assertEquals(new ShardRange(kvColumnStart, kvColumnEnd), plan.keyValueColumns());
        assertEquals(new ShardRange(mlpStart, mlpEnd), plan.mlpIntermediate());
    }

    private static Config gemma2LikeConfig() {
        return new Config(
                8192,
                2304,
                9216,
                8,
                4,
                26,
                1.0e-6f,
                256000,
                2,
                List.of(1, 107),
                ActivationFunction.Type.GELU_PYTORCH_TANH,
                10000.0,
                null,
                null,
                256,
                30.0f,
                50.0f,
                null,
                null,
                null,
                null,
                List.of("Gemma2ForCausalLM")
        );
    }
}
