package io.teknek.deliverance.model.tensorparallel.transport;

import java.util.UUID;

public record CloseSessionRequest(UUID sessionId) {
}
