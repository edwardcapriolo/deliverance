package io.teknek.deliverance.model.tensorparallel;

import io.teknek.deliverance.model.tensorparallel.transport.SharedKvPrefixRestoreRequest;
import io.teknek.deliverance.model.tensorparallel.transport.SharedKvPrefixRestoreResult;
import io.teknek.deliverance.model.tensorparallel.transport.SharedKvPrefixStoreRequest;
import io.teknek.deliverance.model.tensorparallel.transport.TensorParallelRankService;
import io.teknek.deliverance.tensor.AbstractTensor;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;

class TensorParallelGenerationGroupSharedKvPrefixTest {

    @Test
    void restoreSharedKvPrefixReturnsAgreedRankLengthAndPreservesRequestSalt() {
        RecordingRankService rank0 = rank(0, 16);
        RecordingRankService rank1 = rank(1, 16);
        UUID sessionId = UUID.randomUUID();
        int[] promptTokens = {1, 2, 3, 4};

        try (TensorParallelGenerationGroup group = group(List.of(rank0, rank1))) {
            assertEquals(16, group.restoreSharedKvPrefix(sessionId, promptTokens, "request-salt"));
        }

        assertEquals("request-salt", rank0.restored.get().cacheSalt());
        assertEquals("request-salt", rank1.restored.get().cacheSalt());
    }

    @Test
    void restoreSharedKvPrefixFallsBackToZeroWhenRanksDisagreeAndClosesPartialSession() {
        RecordingRankService rank0 = rank(0, 16);
        RecordingRankService rank1 = rank(1, 8);
        UUID sessionId = UUID.randomUUID();

        try (TensorParallelGenerationGroup group = group(List.of(rank0, rank1))) {
            assertEquals(0, group.restoreSharedKvPrefix(sessionId, new int[]{1, 2, 3, 4}, "request-salt"));
        }

        assertEquals(1, rank0.closeCalls.get());
        assertEquals(1, rank1.closeCalls.get());
    }

    @Test
    void storeSharedKvPrefixFansOutToEveryRank() {
        RecordingRankService rank0 = rank(0, 16);
        RecordingRankService rank1 = rank(1, 16);
        UUID sessionId = UUID.randomUUID();
        int[] promptTokens = {1, 2, 3, 4};

        try (TensorParallelGenerationGroup group = group(List.of(rank0, rank1))) {
            group.storeSharedKvPrefix(sessionId, promptTokens, "request-salt");
        }

        assertEquals("request-salt", rank0.stored.get().cacheSalt());
        assertEquals("request-salt", rank1.stored.get().cacheSalt());
    }

    private static TensorParallelGenerationGroup group(List<RecordingRankService> services) {
        List<TensorParallelGenerationGroup.RankEndpoint> endpoints = new ArrayList<>();
        for (RecordingRankService service : services) {
            endpoints.add(new TensorParallelGenerationGroup.RankEndpoint(service.rank, services.size(), service, false));
        }
        return TensorParallelGenerationGroup.fromEndpoints(endpoints);
    }

    private static RecordingRankService rank(int rank, int prefixLength) {
        return new RecordingRankService(rank, prefixLength);
    }

    private static final class RecordingRankService implements TensorParallelRankService {
        private final int rank;
        private final int prefixLength;
        private final AtomicInteger closeCalls = new AtomicInteger();
        private final AtomicReference<SharedKvPrefixRestoreRequest> restored = new AtomicReference<>();
        private final AtomicReference<SharedKvPrefixStoreRequest> stored = new AtomicReference<>();

        private RecordingRankService(int rank, int prefixLength) {
            this.rank = rank;
            this.prefixLength = prefixLength;
        }

        @Override
        public AbstractTensor batchForward(UUID sessionId, int[] tokenIds, int startPosition) {
            throw new UnsupportedOperationException();
        }

        @Override
        public AbstractTensor forward(UUID sessionId, int tokenId, int position) {
            throw new UnsupportedOperationException();
        }

        @Override
        public SharedKvPrefixRestoreResult restoreSharedKvPrefix(SharedKvPrefixRestoreRequest request) {
            restored.set(request);
            return new SharedKvPrefixRestoreResult(prefixLength);
        }

        @Override
        public void storeSharedKvPrefix(SharedKvPrefixStoreRequest request) {
            stored.set(request);
        }

        @Override
        public void closeSession(UUID sessionId) {
            closeCalls.incrementAndGet();
        }
    }
}
