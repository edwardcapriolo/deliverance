package io.teknek.deliverance.model.tensorparallel.transport;

import java.util.UUID;

public record SharedKvPrefixRestoreRequest(UUID sessionId, int[] tokenIds, String cacheSalt) {
}
