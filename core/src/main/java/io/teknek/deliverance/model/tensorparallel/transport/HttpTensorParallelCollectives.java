package io.teknek.deliverance.model.tensorparallel.transport;

import io.teknek.deliverance.JsonUtils;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.model.tensorparallel.TensorParallelCollectives;
import io.teknek.deliverance.model.tensorparallel.TensorParallelContext;
import io.teknek.deliverance.tensor.AbstractTensor;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.concurrent.atomic.AtomicLong;

/**
 * HTTP client implementation of tensor-parallel collectives.
 */
public class HttpTensorParallelCollectives implements TensorParallelCollectives {
    private final HttpClient client = HttpClient.newHttpClient();
    private final TensorParallelContext context;
    private final URI baseUri;
    private final TensorPayloadCodec codec = new BinaryTensorPayloadCodec();
    private final AtomicLong collectiveSequence = new AtomicLong();

    public HttpTensorParallelCollectives(TensorParallelContext context, URI baseUri) {
        this.context = context;
        this.baseUri = baseUri;
    }

    @Override
    public AbstractTensor allReduceSum(String key, AbstractTensor local) {
        String wireKey = collectiveSequence.getAndIncrement() + ":" + key;
        byte[] body = InferenceProfiler.time("collective.client.serialize", () -> {
            try {
                byte[] header = JsonUtils.om.writeValueAsBytes(new AllReduceSumRequest(wireKey, context.rank(), context.size()));
                byte[] tensor = codec.encode(local);
                return ByteBuffer.allocate(Integer.BYTES + header.length + tensor.length)
                        .order(ByteOrder.LITTLE_ENDIAN)
                        .putInt(header.length)
                        .put(header)
                        .put(tensor)
                        .array();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });
        HttpRequest request = HttpRequest.newBuilder(baseUri.resolve("/allReduceSum"))
                .header("Content-Type", "application/octet-stream")
                .POST(HttpRequest.BodyPublishers.ofByteArray(body))
                .build();
        HttpResponse<byte[]> response = InferenceProfiler.time("collective.client.http", () -> {
            try {
                return client.send(request, HttpResponse.BodyHandlers.ofByteArray());
            } catch (IOException e) {
                throw new RuntimeException(e);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException(e);
            }
        });
        if (response.statusCode() != 200) {
            throw new IllegalStateException("Collective server returned HTTP " + response.statusCode());
        }
        return InferenceProfiler.time("collective.client.deserialize", () -> codec.decode(response.body()));
    }
}
