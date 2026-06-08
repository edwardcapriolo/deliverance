package io.teknek.deliverance.model.tensorparallel.transport;

import io.teknek.deliverance.JsonUtils;
import io.teknek.deliverance.tensor.AbstractTensor;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.UUID;

/**
 * Minimal JDK HTTP client for tensor-parallel rank forward operations.
 */
public class HttpTensorParallelRankClient implements TensorParallelRankService {
    private final HttpClient client = HttpClient.newHttpClient();
    private final URI baseUri;
    private final TensorPayloadCodec codec = new BinaryTensorPayloadCodec();

    public HttpTensorParallelRankClient(URI baseUri) {
        this.baseUri = baseUri;
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
            HttpRequest request = HttpRequest.newBuilder(baseUri.resolve(path))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofByteArray(json))
                    .build();
            HttpResponse<byte[]> response = client.send(request, HttpResponse.BodyHandlers.ofByteArray());
            if (response.statusCode() != 200) {
                throw new IllegalStateException("Rank server returned HTTP " + response.statusCode());
            }
            return codec.decode(response.body());
        } catch (IOException e) {
            throw new RuntimeException("HTTP tensor-parallel request failed", e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("HTTP tensor-parallel request interrupted", e);
        }
    }

    private void postNoBody(String path, Object requestBody) {
        try {
            byte[] json = JsonUtils.om.writeValueAsBytes(requestBody);
            HttpRequest request = HttpRequest.newBuilder(baseUri.resolve(path))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofByteArray(json))
                    .build();
            HttpResponse<byte[]> response = client.send(request, HttpResponse.BodyHandlers.ofByteArray());
            if (response.statusCode() != 204) {
                throw new IllegalStateException("Rank server returned HTTP " + response.statusCode());
            }
        } catch (IOException e) {
            throw new RuntimeException("HTTP tensor-parallel request failed", e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("HTTP tensor-parallel request interrupted", e);
        }
    }
}
