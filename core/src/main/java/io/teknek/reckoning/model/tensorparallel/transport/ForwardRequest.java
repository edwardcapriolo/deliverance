package io.teknek.reckoning.model.tensorparallel.transport;

import java.util.UUID;

public record ForwardRequest(UUID sessionId, int tokenId, int position) {
}
