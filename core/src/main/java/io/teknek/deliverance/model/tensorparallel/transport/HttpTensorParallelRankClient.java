package io.teknek.deliverance.model.tensorparallel.transport;

import io.teknek.deliverance.JsonUtils;
import io.teknek.deliverance.tensor.AbstractTensor;

import java.io.IOException;
import java.net.URI;
import java.time.Duration;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.UUID;

/**
 * Minimal JDK HTTP client for tensor-parallel rank forward operations.
 */
public class HttpTensorParallelRankClient implements TensorParallelRankService {
    private final HttpClient client;
    private final URI baseUri;
    private final Duration requestTimeout;
    private final TensorPayloadCodec codec = new BinaryTensorPayloadCodec();

    public HttpTensorParallelRankClient(URI baseUri) {
        this(baseUri, Duration.ofSeconds(5), Duration.ofSeconds(30));
    }

    public HttpTensorParallelRankClient(URI baseUri, Duration connectTimeout, Duration requestTimeout) {
        this.baseUri = baseUri;
        this.requestTimeout = requestTimeout;
        this.client = HttpClient.newBuilder()
                .connectTimeout(connectTimeout)
                .build();
    }

    @Override
    public AbstractTensor batchForward(UUID sessionId, int[] tokenIds, int startPosition) {
        return post("/batchForward", new BatchForwardRequest(sessionId, tokenIds, startPosition));
    }

    @Override
    public AbstractTensor forward(UUID sessionId, int tokenId, int position) {
        return post("/forward", new ForwardRequest(sessionId, tokenId, position));
    }

    @Override
    public void closeSession(UUID sessionId) {
        postNoBody("/closeSession", new CloseSessionRequest(sessionId));
    }

    private AbstractTensor post(String path, Object requestBody) {
        try {
            byte[] json = JsonUtils.om.writeValueAsBytes(requestBody);
            URI uri = baseUri.resolve(path);
            HttpRequest request = HttpRequest.newBuilder(uri)
                    .timeout(requestTimeout)
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofByteArray(json))
                    .build();
            HttpResponse<byte[]> response = client.send(request, HttpResponse.BodyHandlers.ofByteArray());
            if (response.statusCode() != 200) {
                throw new IllegalStateException("Rank server returned HTTP " + response.statusCode()
                        + " uri=" + uri);
            }
            return codec.decode(response.body());
        } catch (IOException e) {
            throw new RuntimeException("HTTP tensor-parallel request failed uri=" + baseUri.resolve(path)
                    + " timeout=" + requestTimeout, e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("HTTP tensor-parallel request interrupted", e);
        }
    }

    private void postNoBody(String path, Object requestBody) {
        try {
            byte[] json = JsonUtils.om.writeValueAsBytes(requestBody);
            URI uri = baseUri.resolve(path);
            HttpRequest request = HttpRequest.newBuilder(uri)
                    .timeout(requestTimeout)
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofByteArray(json))
                    .build();
            HttpResponse<byte[]> response = client.send(request, HttpResponse.BodyHandlers.ofByteArray());
            if (response.statusCode() != 204) {
                throw new IllegalStateException("Rank server returned HTTP " + response.statusCode()
                        + " uri=" + uri);
            }
        } catch (IOException e) {
            throw new RuntimeException("HTTP tensor-parallel request failed uri=" + baseUri.resolve(path)
                    + " timeout=" + requestTimeout, e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("HTTP tensor-parallel request interrupted", e);
        }
    }

}
