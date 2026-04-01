package io.teknek.deliverance.model;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.embedding.PoolingType;
import io.teknek.deliverance.math.VectorMathUtils;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorCache;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

public class TestModels {
    private static final Logger logger = LoggerFactory.getLogger(TestModels.class);

    @Test
    public void LeafModelRun() throws Exception {
        String modelOwner = "MongoDB";
        String modelName = "mdbr-leaf-ir";
        ModelFetcher fetch = new ModelFetcher(modelOwner, modelName);
        File localModelPath = fetch.maybeDownload();

        MetricRegistry mr = new MetricRegistry();
        TensorCache tensorCache = new TensorCache(mr);
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores())) {
            ConfigurableTensorProvider provider = new ConfigurableTensorProvider(tensorCache, pool);
            AbstractModel model = ModelSupport.loadEmbeddingModel(localModelPath, DType.F32, DType.F32, provider,
                    mr, tensorCache, new KvBufferCacheSettings(true));
            Assertions.assertEquals(384, model.getConfig().embeddingLength, "LEAF model should have 384 embedding dimensions");

            String query1 = "What is artificial intelligence?";
            float[] embedding1 = model.embed(query1, PoolingType.AVG);
            Assertions.assertEquals(384, embedding1.length, "Embedding should have 384 dimensions");

            boolean hasNonZero = false;
            boolean allFinite = true;
            for (float v : embedding1) {
                if (v != 0.0f) hasNonZero = true;
                if (!Float.isFinite(v)) allFinite = false;
            }
            Assertions.assertTrue(hasNonZero, "Embedding should have non-zero values");
            Assertions.assertTrue(allFinite, "All embedding values should be finite");

            // Test similarity between related texts
            String query2 = "Define artificial intelligence";
            String query3 = "Which power ranger is the most popular?";

            PoolingType poolingType = PoolingType.AVG;
            float[] embedding2 = model.embed(query2, poolingType);
            float[] embedding3 = model.embed(query3, poolingType);

            float similarity12 = VectorMathUtils.cosineSimilarity(embedding1, embedding2);
            float similarity13 = VectorMathUtils.cosineSimilarity(embedding1, embedding3);

            logger.info("Similarity between '{}' and '{}': {}", query1, query2, String.format("%.4f", similarity12));
            logger.info("Similarity between '{}' and '{}': {}", query1, query3, String.format("%.4f", similarity13));

            // Related queries should have higher similarity than unrelated ones
            Assertions.assertTrue(
                    similarity12 > similarity13,
                    String.format("Related queries should have higher similarity (%.4f > %.4f)", similarity12, similarity13)
            );

            // Test with information retrieval examples
            String base = "MongoDB is a NoSQL database";
            String[] examples = new String[]{
                    "MongoDB stores data in documents",
                    "PostgreSQL is a relational database",
                    "The cat sat on the mat",
                    "NoSQL databases are non-relational",
                    "MongoDB uses BSON format"
            };

            float[] baseEmbedding = model.embed(base, poolingType);
            float maxSimilarity = 0.0f;
            String bestMatch = "";
            int bestIndex = -1;

            for (int i = 0; i < examples.length; i++) {
                float[] exampleEmbedding = model.embed(examples[i], poolingType);
                float similarity = VectorMathUtils.cosineSimilarity(baseEmbedding, exampleEmbedding);
                logger.info("Similarity between '{}' and '{}': {}", base, examples[i], String.format("%.4f", similarity));
                if (similarity > maxSimilarity) {
                    maxSimilarity = similarity;
                    bestMatch = examples[i];
                    bestIndex = i;
                }
            }

            logger.info("Best match for '{}' is '{}' with similarity {}", base, bestMatch, String.format("%.4f", maxSimilarity));

            // The best match should be one of the MongoDB-related examples (indices 0, 3, or 4)
            Assertions.assertTrue(
                    bestIndex == 0 || bestIndex == 3 || bestIndex == 4,
                    "Best match should be MongoDB-related"
            );

            // Performance test
            //long start = System.currentTimeMillis();
            //int iterations = 100;
            //VectorMath.pfor(0, iterations, i -> model.embed(query1, poolingType));
            //long elapsed = System.currentTimeMillis() - start;
            //double avgTime = (double) elapsed / iterations;
            //logger.info("Performance: {} embeddings in {}ms, avg {}ms per embedding", iterations, elapsed, String.format("%.2f", avgTime));

            model.close();
        }
    }
}
