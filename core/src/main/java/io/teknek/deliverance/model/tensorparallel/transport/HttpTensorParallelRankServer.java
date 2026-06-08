package io.teknek.deliverance.model.tensorparallel.transport;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import io.teknek.deliverance.JsonUtils;
import io.teknek.deliverance.tensor.AbstractTensor;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Minimal JDK HTTP server for tensor-parallel rank forward operations.
 */
public class HttpTensorParallelRankServer implements AutoCloseable {
    private final HttpServer server;
    private final ExecutorService executor;
    private final TensorParallelRankService service;
    private final TensorPayloadCodec codec = new BinaryTensorPayloadCodec();

    public HttpTensorParallelRankServer(InetSocketAddress address, TensorParallelRankService service) {
        this.service = service;
        try {
            this.server = HttpServer.create(address, 0);
        } catch (IOException e) {
            throw new RuntimeException("Unable to create tensor-parallel rank HTTP server", e);
        }
        this.executor = Executors.newCachedThreadPool();
        server.setExecutor(executor);
        server.createContext("/batchForward", exchange -> handleBatchForward(exchange, service));
        server.createContext("/forward", exchange -> handleForward(exchange, service));
        server.createContext("/closeSession", exchange -> handleCloseSession(exchange, service));
    }

    public void start() {
        server.start();
    }

    public URI uri() {
        InetSocketAddress address = server.getAddress();
        return URI.create("http://" + address.getHostString() + ":" + address.getPort());
    }

    @Override
    public void close() {
        server.stop(0);
        executor.shutdownNow();
        if (service instanceof AutoCloseable closeable) {
            try {
                closeable.close();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
    }

    private void handleBatchForward(HttpExchange exchange, TensorParallelRankService service) throws IOException {
        BatchForwardRequest request = JsonUtils.om.readValue(exchange.getRequestBody(), BatchForwardRequest.class);
        try (AbstractTensor output = service.batchForward(request.sessionId(), request.tokenIds(), request.startPosition())) {
            writeTensor(exchange, output);
        }
    }

    private void handleForward(HttpExchange exchange, TensorParallelRankService service) throws IOException {
        ForwardRequest request = JsonUtils.om.readValue(exchange.getRequestBody(), ForwardRequest.class);
        try (AbstractTensor output = service.forward(request.sessionId(), request.tokenId(), request.position())) {
            writeTensor(exchange, output);
        }
    }

    private void handleCloseSession(HttpExchange exchange, TensorParallelRankService service) throws IOException {
        CloseSessionRequest request = JsonUtils.om.readValue(exchange.getRequestBody(), CloseSessionRequest.class);
        service.closeSession(request.sessionId());
        exchange.sendResponseHeaders(204, -1);
        exchange.close();
    }

    private void writeTensor(HttpExchange exchange, AbstractTensor tensor) throws IOException {
        byte[] response = codec.encode(tensor);
        exchange.getResponseHeaders().add("Content-Type", codec.contentType());
        exchange.sendResponseHeaders(200, response.length);
        exchange.getResponseBody().write(response);
        exchange.close();
    }
}
