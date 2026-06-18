package io.teknek.deliverance.model.tensorparallel.transport;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.JsonUtils;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import com.codahale.metrics.MetricRegistry;
import com.codahale.metrics.Timer;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.URI;
import java.time.Duration;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Coordinator-hosted HTTP collectives server for tensor-parallel all-reduce operations.
 */
public class HttpTensorParallelCollectiveServer implements AutoCloseable {
    private static final Logger LOGGER = LoggerFactory.getLogger(HttpTensorParallelCollectiveServer.class);
    private static final MetricRegistry METRICS = new MetricRegistry();
    private final HttpServer server;
    private final ExecutorService executor;
    private final Group group;
    private final TensorPayloadCodec codec = new BinaryTensorPayloadCodec();

    public HttpTensorParallelCollectiveServer(InetSocketAddress address, Duration timeout) {
        try {
            this.server = HttpServer.create(address, 0);
        } catch (IOException e) {
            throw new RuntimeException("Unable to create tensor-parallel collective HTTP server", e);
        }
        this.executor = Executors.newCachedThreadPool();
        this.group = new Group(timeout);
        server.setExecutor(executor);
        server.createContext("/allReduceSum", this::handleAllReduceSum);
    }

    public void start() {
        server.start();
        LOGGER.info("Started HTTP tensor-parallel collective server uri={}", uri());
    }

    public URI uri() {
        InetSocketAddress address = server.getAddress();
        return URI.create("http://" + address.getHostString() + ":" + address.getPort());
    }

    @Override
    public void close() {
        LOGGER.info("Closing HTTP tensor-parallel collective server uri={}", uri());
        server.stop(0);
        executor.shutdownNow();
    }

    private void handleAllReduceSum(HttpExchange exchange) throws IOException {
        DecodedRequest decoded;
        try (Timer.Context ignored = InferenceProfiler.timer(METRICS, "collective.server.decode").time()) {
            try {
                byte[] body = exchange.getRequestBody().readAllBytes();
                ByteBuffer bodyBuffer = ByteBuffer.wrap(body).order(ByteOrder.LITTLE_ENDIAN);
                int headerLength = bodyBuffer.getInt();
                if (headerLength < 1 || headerLength > bodyBuffer.remaining()) {
                    throw new IllegalArgumentException("Invalid allReduceSum header length " + headerLength);
                }
                byte[] header = new byte[headerLength];
                bodyBuffer.get(header);
                byte[] tensorBytes = new byte[bodyBuffer.remaining()];
                bodyBuffer.get(tensorBytes);
                AllReduceSumRequest request = JsonUtils.om.readValue(header, AllReduceSumRequest.class);
                decoded = new DecodedRequest(request, codec.decode(tensorBytes));
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
        try (AbstractTensor local = decoded.local();
             AbstractTensor reduced = group.allReduceSum(decoded.request().key(), decoded.request().rank(), decoded.request().size(), local)) {
            byte[] response;
            try (Timer.Context ignored = InferenceProfiler.timer(METRICS, "collective.server.encode").time()) {
                response = codec.encode(reduced);
            }
            exchange.getResponseHeaders().add("Content-Type", codec.contentType());
            exchange.sendResponseHeaders(200, response.length);
            exchange.getResponseBody().write(response);
        } finally {
            exchange.close();
        }
    }

    private record DecodedRequest(AllReduceSumRequest request, AbstractTensor local) {
    }

    private static class Group {
        private final Duration timeout;
        private final Map<String, Round> rounds = new HashMap<>();

        private Group(Duration timeout) {
            this.timeout = timeout;
        }

        private synchronized AbstractTensor allReduceSum(String key, int rank, int size, AbstractTensor local) {
            Round round = rounds.computeIfAbsent(key, ignored -> new Round(size, local));
            round.add(rank, local);
            if (round.arrived == size) {
                try (Timer.Context ignored = InferenceProfiler.timer(METRICS, "collective.server.reduce").time()) {
                    round.reduce();
                }
                notifyAll();
            }
            try (Timer.Context ignored = InferenceProfiler.timer(METRICS, "collective.server.wait").time()) {
                waitForReduction(key, round);
            }
            AbstractTensor result = round.copyReduced();
            round.returned++;
            if (round.returned == size) {
                round.closeReduced();
                rounds.remove(key);
            }
            return result;
        }

        private void waitForReduction(String key, Round round) {
            long deadline = System.nanoTime() + timeout.toNanos();
            while (round.reduced == null) {
                long remainingNanos = deadline - System.nanoTime();
                if (remainingNanos <= 0) {
                    throw new IllegalStateException("Timed out waiting for allReduceSum key=" + key
                            + " expected=" + round.size + " arrived=" + round.arrived);
                }
                try {
                    wait(Math.max(1L, remainingNanos / 1_000_000L));
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new IllegalStateException("Interrupted waiting for allReduceSum key=" + key, e);
                }
            }
        }
    }

    private static class Round {
        private final int size;
        private final int[] shape;
        private final DType dType;
        private final AbstractTensor[] contributions;
        private int arrived;
        private int returned;
        private AbstractTensor reduced;

        private Round(int size, AbstractTensor first) {
            this.size = size;
            this.shape = first.shape().shapeArray();
            this.dType = first.dType();
            this.contributions = new AbstractTensor[size];
        }

        private void add(int rank, AbstractTensor local) {
            if (contributions[rank] != null) {
                throw new IllegalStateException("rank " + rank + " already contributed to this allReduceSum");
            }
            if (local.dType() != dType || !Arrays.equals(local.shape().shapeArray(), shape)) {
                throw new IllegalArgumentException("allReduceSum contributions must have matching dtype and shape");
            }
            contributions[rank] = copy(local);
            arrived++;
        }

        private void reduce() {
            reduced = new FloatBufferTensor(shape);
            for (AbstractTensor contribution : contributions) {
                for (int row = 0; row < reduced.shape().first(); row++) {
                    for (int col = 0; col < reduced.shape().last(); col++) {
                        reduced.set(reduced.get(row, col) + contribution.get(row, col), row, col);
                    }
                }
                contribution.close();
            }
        }

        private AbstractTensor copyReduced() {
            return copy(reduced);
        }

        private void closeReduced() {
            if (reduced != null) {
                reduced.close();
                reduced = null;
            }
        }

        private static AbstractTensor copy(AbstractTensor source) {
            if (source.dType() != DType.F32) {
                throw new UnsupportedOperationException("HTTP collectives currently support F32 tensors");
            }
            AbstractTensor copy = new FloatBufferTensor(source.shape());
            copy.copyFrom(source, 0, 0, (int) source.size());
            return copy;
        }
    }
}
