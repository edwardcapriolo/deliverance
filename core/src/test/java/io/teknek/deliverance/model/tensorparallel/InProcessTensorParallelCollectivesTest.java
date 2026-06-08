package io.teknek.deliverance.model.tensorparallel;

import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorDisplayUtil;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class InProcessTensorParallelCollectivesTest {

    @Test
    public void allReduceSumWaitsForBothRanksAndReturnsReducedCopies() throws Exception {
        InProcessTensorParallelCollectives.Group group = new InProcessTensorParallelCollectives.Group(Duration.ofSeconds(5));
        InProcessTensorParallelCollectives rank0 = new InProcessTensorParallelCollectives(new StaticTensorParallelContext(0, 2), group);
        InProcessTensorParallelCollectives rank1 = new InProcessTensorParallelCollectives(new StaticTensorParallelContext(1, 2), group);

        try (ExecutorService executor = Executors.newFixedThreadPool(2)) {
            Future<AbstractTensor> first = executor.submit(() -> rank0.allReduceSum("test", tensor(1, 2)));
            Future<AbstractTensor> second = executor.submit(() -> rank1.allReduceSum("test", tensor(10, 20)));

            try (AbstractTensor firstResult = first.get(); AbstractTensor secondResult = second.get()) {
                String expected = "[0][0]= 11.0000 [0][1]= 22.0000";
                assertEquals(expected, TensorDisplayUtil.pretty2dDisplayAll(firstResult).trim());
                assertEquals(expected, TensorDisplayUtil.pretty2dDisplayAll(secondResult).trim());
            }
        }
    }

    private static AbstractTensor tensor(float first, float second) {
        AbstractTensor tensor = new FloatBufferTensor(1, 2);
        tensor.set(first, 0, 0);
        tensor.set(second, 0, 1);
        return tensor;
    }
}
