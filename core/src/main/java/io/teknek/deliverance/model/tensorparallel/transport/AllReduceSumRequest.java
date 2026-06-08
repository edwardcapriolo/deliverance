package io.teknek.deliverance.model.tensorparallel.transport;

public record AllReduceSumRequest(String key, int rank, int size) {
}
