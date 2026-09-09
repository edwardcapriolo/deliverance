package io.teknek.deliverance.model.tensorparallel.transport;

import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorDisplayUtil;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.api.Test;

import java.net.InetSocketAddress;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class HttpTensorParallelRankTransportTest {

    @Test
    public void clientCallsBatchAndSingleTokenForwardOverHttp() {
        AtomicReference<UUID> closedSession = new AtomicReference<>();
        AtomicReference<SharedKvPrefixRestoreRequest> restoredPrefix = new AtomicReference<>();
        AtomicReference<SharedKvPrefixStoreRequest> storedPrefix = new AtomicReference<>();
        TensorParallelRankService service = new TensorParallelRankService() {
            @Override
            public AbstractTensor batchForward(UUID sessionId, int[] tokenIds, int startPosition) {
                AbstractTensor tensor = new FloatBufferTensor(1, 3);
                tensor.set(tokenIds.length, 0, 0);
                tensor.set(startPosition, 0, 1);
                tensor.set(tokenIds[0], 0, 2);
                return tensor;
            }

            @Override
            public AbstractTensor forward(UUID sessionId, int tokenId, int position) {
                AbstractTensor tensor = new FloatBufferTensor(1, 3);
                tensor.set(tokenId, 0, 0);
                tensor.set(position, 0, 1);
                tensor.set(99.0f, 0, 2);
                return tensor;
            }

            @Override
            public void closeSession(UUID sessionId) {
                closedSession.set(sessionId);
            }

            @Override
            public SharedKvPrefixRestoreResult restoreSharedKvPrefix(SharedKvPrefixRestoreRequest request) {
                restoredPrefix.set(request);
                return new SharedKvPrefixRestoreResult(8);
            }

            @Override
            public void storeSharedKvPrefix(SharedKvPrefixStoreRequest request) {
                storedPrefix.set(request);
            }
        };

        try (HttpTensorParallelRankServer server = new HttpTensorParallelRankServer(
                new InetSocketAddress("127.0.0.1", 0), service)) {
            server.start();
            HttpTensorParallelRankClient client = new HttpTensorParallelRankClient(server.uri());

            try (AbstractTensor batch = client.batchForward(UUID.randomUUID(), new int[]{7, 8}, 3)) {
                assertEquals("[0][0]=  2.0000 [0][1]=  3.0000 [0][2]=  7.0000".trim(),
                        TensorDisplayUtil.pretty2dDisplayAll(batch).trim());
            }
            try (AbstractTensor single = client.forward(UUID.randomUUID(), 42, 5)) {
                assertEquals("[0][0]= 42.0000 [0][1]=  5.0000 [0][2]= 99.0000".trim(),
                        TensorDisplayUtil.pretty2dDisplayAll(single).trim());
            }
            UUID sessionId = UUID.randomUUID();
            SharedKvPrefixRestoreResult restore = client.restoreSharedKvPrefix(
                    new SharedKvPrefixRestoreRequest(sessionId, new int[]{1, 2, 3, 4}, "salt"));
            assertEquals(8, restore.prefixLength());
            assertEquals(sessionId, restoredPrefix.get().sessionId());
            assertEquals("salt", restoredPrefix.get().cacheSalt());
            client.storeSharedKvPrefix(new SharedKvPrefixStoreRequest(sessionId, new int[]{1, 2, 3, 4}, "salt"));
            assertEquals(sessionId, storedPrefix.get().sessionId());
            assertEquals("salt", storedPrefix.get().cacheSalt());
            client.closeSession(sessionId);
            assertEquals(sessionId, closedSession.get());
        }
    }
}
