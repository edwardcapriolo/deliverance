package io.teknek.deliverance.model.tensorparallel.transport;

import java.util.UUID;

public record ForwardRequest(UUID sessionId, int tokenId, int position) {
}
