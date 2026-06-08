package io.teknek.deliverance.model.tensorparallel.transport;

import io.teknek.deliverance.model.tensorparallel.TensorParallelGenerationGroup;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorDisplayUtil;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.api.Test;

import java.net.InetSocketAddress;
import java.util.List;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class HttpTensorParallelGenerationGroupTest {

    @Test
    public void generationGroupCoordinatesHttpRankClients() {
        try (HttpTensorParallelRankServer rank0Server = new HttpTensorParallelRankServer(
                new InetSocketAddress("127.0.0.1", 0), service(0));
             HttpTensorParallelRankServer rank1Server = new HttpTensorParallelRankServer(
                     new InetSocketAddress("127.0.0.1", 0), service(1))) {
            rank0Server.start();
            rank1Server.start();

            TensorParallelGenerationGroup group = TensorParallelGenerationGroup.fromEndpoints(List.of(
                    new TensorParallelGenerationGroup.RankEndpoint(0, 2,
                            new HttpTensorParallelRankClient(rank0Server.uri()), false),
                    new TensorParallelGenerationGroup.RankEndpoint(1, 2,
                            new HttpTensorParallelRankClient(rank1Server.uri()), false)
            ));

            try (group;
                 AbstractTensor rankZero = group.batchForward(new int[]{4, 5}, 7)) {
                assertEquals("[0][0]=  0.0000 [0][1]=  2.0000 [0][2]=  7.0000".trim(),
                        TensorDisplayUtil.pretty2dDisplayAll(rankZero).trim());
            }
        }
    }

    private static TensorParallelRankService service(int rank) {
        return new TensorParallelRankService() {
            @Override
            public AbstractTensor batchForward(UUID sessionId, int[] tokenIds, int startPosition) {
                AbstractTensor tensor = new FloatBufferTensor(1, 3);
                tensor.set(rank, 0, 0);
                tensor.set(tokenIds.length, 0, 1);
                tensor.set(startPosition, 0, 2);
                return tensor;
            }

            @Override
            public AbstractTensor forward(UUID sessionId, int tokenId, int position) {
                AbstractTensor tensor = new FloatBufferTensor(1, 3);
                tensor.set(rank, 0, 0);
                tensor.set(tokenId, 0, 1);
                tensor.set(position, 0, 2);
                return tensor;
            }

            @Override
            public void closeSession(UUID sessionId) {
            }
        };
    }
}
