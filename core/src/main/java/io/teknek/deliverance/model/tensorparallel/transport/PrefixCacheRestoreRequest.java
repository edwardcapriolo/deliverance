package io.teknek.deliverance.model.tensorparallel.transport;

import java.util.UUID;

public record PrefixCacheRestoreRequest(UUID sessionId, int[] tokenIds, String cacheSalt, int prefixLength) {
}
