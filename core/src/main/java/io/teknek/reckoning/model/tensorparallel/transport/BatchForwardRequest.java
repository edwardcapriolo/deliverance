package io.teknek.reckoning.model.tensorparallel.transport;

import java.util.UUID;

public record BatchForwardRequest(UUID sessionId, int[] tokenIds, int startPosition) {
}
