package io.teknek.reckoning.model.tensorparallel.transport;

public record AllReduceSumRequest(String key, int rank, int size) {
}
