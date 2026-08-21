package io.teknek.deliverance.model.tensorparallel.transport;

public record PrefixCacheProbeRequest(int[] tokenIds, String cacheSalt) {
}
