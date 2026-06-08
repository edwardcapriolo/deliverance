package io.teknek.deliverance.model.tensorparallel;

import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertSame;

public class SingleRankTensorParallelCollectivesTest {

    @Test
    public void allReduceSumReturnsOnlyRankContribution() {
        SingleRankTensorParallelCollectives collectives = new SingleRankTensorParallelCollectives();
        try (AbstractTensor local = new FloatBufferTensor(1, 2)) {
            assertSame(local, collectives.allReduceSum("layer.0.test", local));
        }
    }
}
