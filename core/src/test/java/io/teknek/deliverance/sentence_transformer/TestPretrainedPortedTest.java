package io.teknek.deliverance.sentence_transformer;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.embedding.PoolingType;
import io.teknek.deliverance.math.VectorMathUtils;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForEmbeddings;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.Arrays;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Ports selected cases from sentence-transformers/tests/sentence_transformer/test_pretrained.py. */
class TestPretrainedPortedTest {
    private static final String QUERY = "Which planet is known as the Red Planet?";
    private static final String[] DOCUMENTS = {
            "Venus is often called Earth's twin because of its similar size and proximity.",
            "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
            "Jupiter, the largest planet in our solar system, has a prominent red spot.",
            "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."
    };

    @ParameterizedTest
    @MethodSource("modelsToSimilaritiesBf16Sdpa")
    void testPretrainedModelBf16Sdpa(ModelCase modelCase) {
        assertPretrainedSimilarities(modelCase.owner(), modelCase.modelName(), modelCase.expected());
    }

    private static Stream<ModelCase> modelsToSimilaritiesBf16Sdpa() {
        return Stream.of(
                new ModelCase("BAAI", "bge-small-en-v1.5",
                        new float[] { 0.60191f, 0.82845f, 0.7786f, 0.70781f }),
                new ModelCase("intfloat", "e5-small-v2",
                        new float[] { 0.8147f, 0.91502f, 0.86984f, 0.87874f }),
                new ModelCase("sentence-transformers", "all-MiniLM-L6-v2",
                        new float[] { 0.46371f, 0.81205f, 0.72828f, 0.75051f }));
    }

    @Test
    @Disabled("Requires XLMRobertaTokenizer support; keep the upstream case visible until tokenizer support is added.")
    void testPretrainedModelBf16SdpaMultilingualE5Small() {
        assertPretrainedSimilarities("intfloat", "multilingual-e5-small",
                new float[] { 0.81157f, 0.90596f, 0.87089f, 0.85667f });
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
                float[] similarities = new float[DOCUMENTS.length];
                for (int i = 0; i < DOCUMENTS.length; i++) {
                    float[] documentEmbedding = model.embed(DOCUMENTS[i], PoolingType.AVG);
                    similarities[i] = VectorMathUtils.cosineSimilarity(queryEmbedding, documentEmbedding);
                }
                for (int i = 0; i < DOCUMENTS.length; i++) {
                    assertTrue(Math.abs(similarities[i] - expected[i]) <= Math.abs(expected[i]) * 0.01f,
                            "Expected similarity for " + owner + "/" + modelName + " to be close to "
                                    + Arrays.toString(expected) + ", but got " + Arrays.toString(similarities));
                }
            }
        }
    }

    private record ModelCase(String owner, String modelName, float[] expected) {
        @Override
        public String toString() {
            return owner + "/" + modelName;
        }
    }
}
