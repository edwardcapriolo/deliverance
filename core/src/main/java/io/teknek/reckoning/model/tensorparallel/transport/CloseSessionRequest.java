package io.teknek.reckoning.model.tensorparallel.transport;

import java.util.UUID;

public record CloseSessionRequest(UUID sessionId) {
}
