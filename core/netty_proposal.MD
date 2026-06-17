# Netty Tensor-Parallel Transport Proposal

## Context

Current tensor-parallel collectives use HTTP:

* `HttpTensorParallelCollectives`
* `HttpTensorParallelCollectiveServer`

This was a good correctness-first choice because it avoided Spring Boot and kept the transport easy to test. Profiling now
shows that HTTP request/response overhead is material for decode-time tensor parallelism. In one benchmark run,
`collective.client.http` dominated collective cost while `collective.server.reduce` was small.

The issue is not that localhost HTTP is slow for normal service calls. The issue is that tensor parallelism performs many
small collectives per generated token, so per-request overhead accumulates quickly.

## Goal

Add a lower-overhead collective transport for high-frequency tensor-parallel all-reduce operations.

The proposal is Netty with a custom binary protocol over persistent TCP channels. This is not Netty HTTP.

## Non-Goals

* Do not replace Spring Boot APIs.
* Do not replace rank forward transport initially.
* Do not remove the existing HTTP transport.
* Do not implement a benchmark-only transport.
* Do not use UDP for tensor payloads; tensor messages exceed practical datagram sizes and require reliability/order.

## Transport Choice

Use Netty custom binary protocol, not Netty HTTP.

Reasons:

* persistent TCP connections
* compact binary frames
* async I/O and backpressure
* less per-collective allocation/request overhead
* avoids HTTP headers and request routing for every all-reduce
* common fit for high-frequency data-plane protocols

HTTP should remain available for correctness, diagnostics, and simpler deployments.

## Pluggable Shape

Keep the existing conceptual seam:

```java
TensorParallelCollectives
```

Add a collective-server abstraction:

```java
public interface TensorParallelCollectiveServer extends AutoCloseable {
    void start();
    URI uri();
    TensorParallelTransport transport();
}
```

Transport enum:

```java
public enum TensorParallelTransport {
    HTTP,
    NETTY
}
```

Factory sketch:

```java
public final class TensorParallelTransportFactory {
    public static TensorParallelCollectiveServer collectiveServer(
            TensorParallelTransport transport,
            InetSocketAddress bindAddress,
            Duration timeout) {
        return switch (transport) {
            case HTTP -> new HttpTensorParallelCollectiveServer(bindAddress, timeout);
            case NETTY -> new NettyTensorParallelCollectiveServer(bindAddress, timeout);
        };
    }

    public static TensorParallelCollectives collectives(
            TensorParallelTransport transport,
            TensorParallelContext context,
            URI uri) {
        return switch (transport) {
            case HTTP -> new HttpTensorParallelCollectives(context, uri);
            case NETTY -> new NettyTensorParallelCollectives(context, uri);
        };
    }
}
```

URI schemes can carry transport in gossip payloads:

```text
http://127.0.0.1:12345
netty://127.0.0.1:12345
```

This keeps gossip payloads simple and lets consumers infer the transport from URI scheme.

## Protocol Sketch

Implement only `allReduceSum` first.

Frame fields:

```text
magic
version
operation
sequence
key_length
key_bytes
rank
size
dtype
dims
dim0 ... dimN
payload_length
payload_bytes
```

Server behavior:

1. Read contribution frame from each rank.
2. Group by collective key and sequence.
3. Validate rank, size, dtype, and shape.
4. Reduce when all ranks arrive.
5. Write reduced tensor frame back to each waiting rank channel.

## Expected Benefit

This targets measured overhead in `collective.client.http`.

Expected improvements come from removing:

* HTTP request/response object lifecycle per collective
* HTTP headers per collective
* route/context dispatch per collective
* repeated request body publisher/handler setup

The exact gain must be measured. Netty should not be assumed to fix TP by itself.

## Rollout Plan

1. Keep HTTP as default.
2. Add Netty transport behind an explicit setting.
3. Implement only `allReduceSum` over Netty.
4. Compare HTTP vs Netty using the same tensor-parallel benchmark.
5. If Netty materially reduces collective overhead, consider moving rank forward transport later.

## Open Questions

* Should the collective transport setting live in `GossipParallelSettings` or a narrower transport settings object?
* Should Netty use one channel per rank or multiplex multiple rank streams over one channel?
* How should failures and rank disconnects propagate to generation callers?
* How much tensor copy overhead remains after replacing HTTP?
