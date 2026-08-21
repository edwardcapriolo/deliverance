package io.teknek.deliverance.model.tensorparallel;

import io.teknek.deliverance.model.tensorparallel.transport.PrefixCacheProbeRequest;
import io.teknek.deliverance.model.tensorparallel.transport.PrefixCacheProbeResult;
import io.teknek.deliverance.model.tensorparallel.transport.PrefixCacheRestoreRequest;
import io.teknek.deliverance.model.tensorparallel.transport.PrefixCacheRestoreResult;
import io.teknek.deliverance.model.tensorparallel.transport.PrefixCacheStoreRequest;
import io.teknek.deliverance.model.tensorparallel.transport.TensorParallelRankService;
import io.teknek.deliverance.tensor.AbstractTensor;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TensorParallelGenerationGroupPrefixCacheTest {

    @Test
    void probePrefixReturnsAgreedLengthWhenAllRanksHit() {
        try (TensorParallelGenerationGroup group = group(List.of(rank(0, true, 32), rank(1, true, 32)))) {
            assertEquals(32, group.probePrefix(new int[]{1, 2, 3}, "salt"));
        }
    }

    @Test
    void probePrefixMissesWhenAnyRankMisses() {
        try (TensorParallelGenerationGroup group = group(List.of(rank(0, true, 32), rank(1, false, 0)))) {
            assertEquals(0, group.probePrefix(new int[]{1, 2, 3}, "salt"));
        }
    }

    @Test
    void probePrefixMissesWhenRanksDisagreeOnLength() {
        try (TensorParallelGenerationGroup group = group(List.of(rank(0, true, 32), rank(1, true, 64)))) {
            assertEquals(0, group.probePrefix(new int[]{1, 2, 3}, "salt"));
        }
    }

    @Test
    void restorePrefixRequiresEveryRankToRestoreRequestedLength() {
        try (TensorParallelGenerationGroup group = group(List.of(rank(0, true, 32), rank(1, true, 16)))) {
            assertFalse(group.restorePrefix(UUID.randomUUID(), new int[]{1, 2, 3}, "salt", 32));
        }
    }

    @Test
    void storePrefixFansOutToEveryRank() {
        CountingRankService rank0 = rank(0, true, 32);
        CountingRankService rank1 = rank(1, true, 32);
        try (TensorParallelGenerationGroup group = group(List.of(rank0, rank1))) {
            group.storePrefix(UUID.randomUUID(), new int[]{1, 2, 3}, "salt");

            assertEquals(1, rank0.storeCalls.get());
            assertEquals(1, rank1.storeCalls.get());
        }
    }

    private static TensorParallelGenerationGroup group(List<CountingRankService> services) {
        List<TensorParallelGenerationGroup.RankEndpoint> endpoints = new ArrayList<>();
        for (CountingRankService service : services) {
            endpoints.add(new TensorParallelGenerationGroup.RankEndpoint(service.rank, services.size(), service, false));
        }
        return TensorParallelGenerationGroup.fromEndpoints(endpoints);
    }

    private static CountingRankService rank(int rank, boolean hit, int prefixLength) {
        return new CountingRankService(rank, hit, prefixLength);
    }

    private static final class CountingRankService implements TensorParallelRankService {
        private final int rank;
        private final boolean hit;
        private final int prefixLength;
        private final AtomicInteger storeCalls = new AtomicInteger();

        private CountingRankService(int rank, boolean hit, int prefixLength) {
            this.rank = rank;
            this.hit = hit;
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
        public PrefixCacheProbeResult probePrefix(PrefixCacheProbeRequest request) {
            return new PrefixCacheProbeResult(hit, prefixLength);
        }

        @Override
        public PrefixCacheRestoreResult restorePrefix(PrefixCacheRestoreRequest request) {
            return new PrefixCacheRestoreResult(hit && prefixLength == request.prefixLength(), prefixLength);
        }

        @Override
        public void storePrefix(PrefixCacheStoreRequest request) {
            storeCalls.incrementAndGet();
        }

        @Override
        public void closeSession(UUID sessionId) {
        }
    }
}
