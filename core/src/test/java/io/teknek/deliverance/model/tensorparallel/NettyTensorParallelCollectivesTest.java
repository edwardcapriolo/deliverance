package io.teknek.deliverance.model.tensorparallel;

import io.teknek.deliverance.model.tensorparallel.transport.NettyTensorParallelCollectiveServer;
import io.teknek.deliverance.model.tensorparallel.transport.NettyTensorParallelCollectives;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.api.Test;

import java.net.InetSocketAddress;
import java.time.Duration;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class NettyTensorParallelCollectivesTest {

    @Test
    public void allReduceSumOverNettyReturnsReducedCopiesToBothRanks() throws Exception {
        try (NettyTensorParallelCollectiveServer server = new NettyTensorParallelCollectiveServer(
                new InetSocketAddress("127.0.0.1", 0), Duration.ofSeconds(5))) {
            server.start();
            try (NettyTensorParallelCollectives rank0 = new NettyTensorParallelCollectives(
                    new StaticTensorParallelContext(0, 2), server.uri());
                 NettyTensorParallelCollectives rank1 = new NettyTensorParallelCollectives(
                         new StaticTensorParallelContext(1, 2), server.uri());
                 ExecutorService executor = Executors.newFixedThreadPool(2)) {
                Future<AbstractTensor> first = executor.submit(() -> rank0.allReduceSum("test", tensor(1, 2)));
                Future<AbstractTensor> second = executor.submit(() -> rank1.allReduceSum("test", tensor(10, 20)));

                try (AbstractTensor firstResult = first.get(); AbstractTensor secondResult = second.get()) {
                    assertReduced(firstResult);
                    assertReduced(secondResult);
                }
            }
        }
    }

    private static void assertReduced(AbstractTensor tensor) {
        assertEquals(11.0f, tensor.get(0, 0));
        assertEquals(22.0f, tensor.get(0, 1));
    }

    private static AbstractTensor tensor(float first, float second) {
        AbstractTensor tensor = new FloatBufferTensor(1, 2);
        tensor.set(first, 0, 0);
        tensor.set(second, 0, 1);
        return tensor;
    }
}
