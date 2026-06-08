package io.teknek.deliverance.model.tensorparallel.transport;

import io.teknek.deliverance.JsonUtils;
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
        try {
            String wireKey = collectiveSequence.getAndIncrement() + ":" + key;
            byte[] header = JsonUtils.om.writeValueAsBytes(new AllReduceSumRequest(wireKey, context.rank(), context.size()));
            byte[] tensor = codec.encode(local);
            byte[] body = ByteBuffer.allocate(Integer.BYTES + header.length + tensor.length)
                    .order(ByteOrder.LITTLE_ENDIAN)
                    .putInt(header.length)
                    .put(header)
                    .put(tensor)
                    .array();
            HttpRequest request = HttpRequest.newBuilder(baseUri.resolve("/allReduceSum"))
                    .header("Content-Type", "application/octet-stream")
                    .POST(HttpRequest.BodyPublishers.ofByteArray(body))
                    .build();
            HttpResponse<byte[]> response = client.send(request, HttpResponse.BodyHandlers.ofByteArray());
            if (response.statusCode() != 200) {
                throw new IllegalStateException("Collective server returned HTTP " + response.statusCode());
            }
            return codec.decode(response.body());
        } catch (IOException e) {
            throw new RuntimeException("HTTP allReduceSum failed", e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("HTTP allReduceSum interrupted", e);
        }
    }
}
