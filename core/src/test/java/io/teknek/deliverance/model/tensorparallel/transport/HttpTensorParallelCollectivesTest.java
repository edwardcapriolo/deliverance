package io.teknek.deliverance.model.tensorparallel.transport;

import io.teknek.deliverance.model.tensorparallel.StaticTensorParallelContext;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorDisplayUtil;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.api.Test;

import java.net.InetSocketAddress;
import java.time.Duration;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class HttpTensorParallelCollectivesTest {

    @Test
    public void allReduceSumCoordinatesTwoRanksOverHttp() throws Exception {
        try (HttpTensorParallelCollectiveServer server = new HttpTensorParallelCollectiveServer(
                new InetSocketAddress("127.0.0.1", 0), Duration.ofSeconds(5))) {
            server.start();
            HttpTensorParallelCollectives rank0 = new HttpTensorParallelCollectives(new StaticTensorParallelContext(0, 2), server.uri());
            HttpTensorParallelCollectives rank1 = new HttpTensorParallelCollectives(new StaticTensorParallelContext(1, 2), server.uri());
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
    }

    private static AbstractTensor tensor(float first, float second) {
        AbstractTensor tensor = new FloatBufferTensor(1, 2);
        tensor.set(first, 0, 0);
        tensor.set(second, 0, 1);
        return tensor;
    }
}
