package io.teknek.deliverance.embedding;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.VectorMathUtils;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForEmbeddings;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** Ports selected cases from sentence-transformers/tests/sentence_transformer/test_pretrained.py. */
class SentenceTransformersTestPretrainedPortedTest {
    private static final String QUERY = "Which planet is known as the Red Planet?";
    private static final String[] DOCUMENTS = {
            "Venus is often called Earth's twin because of its similar size and proximity.",
            "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
            "Jupiter, the largest planet in our solar system, has a prominent red spot.",
            "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."
    };

    @Test
    @Tag("large-model")
    void testPretrainedModelBf16SdpaAllMiniLmL6V2() {
        assertPretrainedSimilarities("sentence-transformers", "all-MiniLM-L6-v2",
                new float[] { 0.46371f, 0.81205f, 0.72828f, 0.75051f });
    }

    private static void assertPretrainedSimilarities(String owner, String modelName, float[] expected) {
        ModelFetcher fetch = new ModelFetcher(owner, modelName);
        MetricRegistry metrics = new MetricRegistry();
        ArrayQueueTensorAllocator allocator = new ArrayQueueTensorAllocator(metrics);
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores())) {
            AutoModelForEmbeddings.Builder builder = AutoModelForEmbeddings.newBuilder(fetch);
            builder.withWorkingMemoryType(DType.F32);
            builder.withWorkingQuantType(DType.F32);
            builder.withMetricRegistry(metrics);
            builder.withTensorAllocator(allocator);
            builder.withTensorProvider(new ConfigurableTensorProvider(allocator, pool));
            builder.withKvBufferCacheSettings(new KvBufferCacheSettings(true));
            builder.withWrappedForkJoinPool(pool);
            try (AbstractModel model = builder.buildLocalEmbeddingModel()) {
            float[] queryEmbedding = model.embed(QUERY, PoolingType.AVG);
            assertEquals(384, queryEmbedding.length, "Embedding should have 384 dimensions");
            for (int i = 0; i < DOCUMENTS.length; i++) {
                float[] documentEmbedding = model.embed(DOCUMENTS[i], PoolingType.AVG);
                float similarity = VectorMathUtils.cosineSimilarity(queryEmbedding, documentEmbedding);
                assertEquals(expected[i], similarity, 0.01f, "document=" + i);
            }
            }
        }
    }
}
