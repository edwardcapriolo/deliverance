package io.teknek.deliverance.embedding;

import com.fasterxml.jackson.databind.JsonNode;
import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.JsonUtils;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.ModelSupport;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.InputStream;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/** Real-checkpoint parity gate for one fixed EmbeddingGemma 2 Python reference result. */
class EmbeddingGemma2ReferenceParityTest {
    private static final String TEXT = "This is a test document about machine learning";
    @Test
    @Tag("large-model")
    void matchesPythonReferenceForKnownDocument() throws Exception {
        ModelFetcher fetcher = new ModelFetcher("google", "embeddinggemma-2");
        File modelDirectory = fetcher.maybeDownload();
        MetricRegistry metrics = new MetricRegistry();
        ArrayQueueTensorAllocator allocator = new ArrayQueueTensorAllocator(metrics);
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores());
             AbstractModel model = ModelSupport.loadEmbeddingModel(modelDirectory, DType.F32, DType.F32,
                     new ConfigurableTensorProvider(allocator, pool), metrics, allocator,
                     new KvBufferCacheSettings(true))) {
            int[] tokenIds = model.getTokenizer().encode(TEXT).inputIds();
            JsonNode reference = referenceFixture();
            assertEquals(reference.get("text").asText(), TEXT);
            int[] expectedTokenIds = JsonUtils.om.convertValue(reference.get("tokenIds"), int[].class);
            assertArrayEquals(expectedTokenIds, tokenIds, Arrays.toString(tokenIds));
            float[] embedding = model.embed(TEXT, PoolingType.AVG);
            float[] expectedEmbedding = JsonUtils.om.convertValue(reference.get("embedding"), float[].class);
            assertEquals(expectedEmbedding.length, embedding.length);
            assertEquals(reference.get("dimension").asInt(), embedding.length);
            for (int index = 0; index < expectedEmbedding.length; index++) {
                assertEquals(expectedEmbedding[index], embedding[index], 1.0e-3f,
                        "embedding index=" + index);
            }
            double norm = 0.0;
            for (float value : embedding) norm += (double) value * value;
            assertEquals(reference.get("l2Norm").asDouble(), Math.sqrt(norm), 1.0e-3);
        }
    }

    private static JsonNode referenceFixture() throws Exception {
        InputStream stream = EmbeddingGemma2ReferenceParityTest.class.getResourceAsStream(
                "/embed_gemma2.json");
        if (stream == null) {
            throw new IllegalStateException("Missing embed_gemma2.json; generate it with "
                    + "core/scripts/embedding_gemma2_reference.py");
        }
        try (stream) {
            return JsonUtils.om.readTree(stream);
        }
    }
}
