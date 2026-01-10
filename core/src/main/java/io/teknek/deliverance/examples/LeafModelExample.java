/*
 * Copyright 2025 Tiernan Lindauer
 *
 * Licensed under the Apache License, version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */
package io.teknek.deliverance.examples;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.embedding.PoolingType;
import io.teknek.deliverance.math.VectorMathUtils;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.ModelSupport;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorCache;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.stream.IntStream;

/**
 * Example demonstrating how to use the LEAF embedding model (MongoDB/mdbr-leaf-ir) with Deliverance.
 *
 * Improvements over a basic example:
 *  - Robust error handling with clear messages and exit codes
 *  - Optional L2 normalization of embeddings
 *  - Batch-embedding helper (simple loop; toggleable parallel mode)
 *  - Uses System.nanoTime() for accurate timing
 *  - CLI flags for working directory, iterations, batch size, normalization, and parallel embedding
 *
 * Usage:
 *   java -cp ... io.teknek.deliverance.examples.LeafModelExample [--model-dir DIR] [--iterations N]
 *         [--batch-size N] [--normalize true|false] [--parallel true|false]
 *
 * Example:
 *   java ... LeafModelExample --model-dir ./models --iterations 200 --batch-size 8 --normalize true
 */
public class LeafModelExample {
    private static final String DEFAULT_MODEL_OWNER = "MongoDB";
    private static final String DEFAULT_MODEL_NAME = "mdbr-leaf-ir";
    private static final int DEFAULT_ITERATIONS = 100;
    private static final int DEFAULT_BATCH_SIZE = 1;
    private static final boolean DEFAULT_NORMALIZE = false;
    private static final boolean DEFAULT_PARALLEL = false;

    public static void main(String[] args) {
        // Simple CLI parsing
        String modelOwner = DEFAULT_MODEL_OWNER;
        String modelName = DEFAULT_MODEL_NAME;
        int iterations = DEFAULT_ITERATIONS;
        int batchSize = DEFAULT_BATCH_SIZE;
        boolean l2Normalize = DEFAULT_NORMALIZE;
        boolean parallel = DEFAULT_PARALLEL;

        try {
            for (int i = 0; i < args.length; i++) {
                String a = args[i];
                if ("--iterations".equalsIgnoreCase(a) && i + 1 < args.length) {
                    iterations = Integer.parseInt(args[++i]);
                } else if ("--batch-size".equalsIgnoreCase(a) && i + 1 < args.length) {
                    batchSize = Integer.parseInt(args[++i]);
                } else if ("--normalize".equalsIgnoreCase(a) && i + 1 < args.length) {
                    l2Normalize = Boolean.parseBoolean(args[++i]);
                } else if ("--parallel".equalsIgnoreCase(a) && i + 1 < args.length) {
                    parallel = Boolean.parseBoolean(args[++i]);
                } else if ("--help".equalsIgnoreCase(a) || "-h".equalsIgnoreCase(a)) {
                    printUsageAndExit(0);
                } else {
                    // ignored unknown flags for forward-compatibility
                }
            }
        } catch (Exception e) {
            System.err.println("Error parsing arguments: " + e.getMessage());
            printUsageAndExit(2);
        }

        System.out.println("=== LEAF Model Integration Test (improved) ===");
        System.out.println("Model: " + modelOwner + "/" + modelName);
        System.out.println("Iterations: " + iterations);
        System.out.println("Batch size: " + batchSize);
        System.out.println("L2-normalize: " + l2Normalize);
        System.out.println("Parallel embedding: " + parallel);
        System.out.println();

        // Main flow wrapped in try/catch so we can exit cleanly on errors
        try {
            // Download or find model
            System.out.println("Downloading or locating LEAF model...");
            ModelFetcher fetch = new ModelFetcher(modelOwner, modelName);
            File localModelPath = fetch.maybeDownload();
            if (localModelPath == null || !localModelPath.exists()) {
                System.err.println("Failed to locate or download model");
                System.exit(3);
            }
            System.out.println("Model located at: " + localModelPath.getAbsolutePath());
            System.out.println();

            // Load embedding model (use F32 as the default dtype for LEAF)
            System.out.println("Loading model...");
            MetricRegistry mr = new MetricRegistry();
            TensorCache tensorCache = new TensorCache(mr);
            ConfigurableTensorProvider provider = new ConfigurableTensorProvider(tensorCache);
            AbstractModel model = ModelSupport.loadEmbeddingModel(localModelPath, DType.F32, DType.F32, provider,
                    mr, tensorCache, new KvBufferCacheSettings(true));
            if (model == null) {
                System.err.println("ModelSupport.loadEmbeddingModel returned null - cannot proceed.");
                System.exit(4);
            }
            System.out.println("Model loaded successfully!");
            int embeddingLength = model.getConfig().embeddingLength;
            System.out.println("Embedding dimensions reported by model: " + embeddingLength);
            System.out.println("Expected (LEAF): 384");
            if (embeddingLength != 384) {
                System.err.println("WARNING: Expected 384 dimensions but got " + embeddingLength);
            }
            System.out.println();

            // Samples for tests
            String query1 = "What is artificial intelligence?";
            String query2 = "Define artificial intelligence";
            String query3 = "What is the weather today?";
            List<String> batchInputs = Arrays.asList(
                    "MongoDB stores data in documents",
                    "PostgreSQL is a relational database",
                    "The cat sat on the mat",
                    "NoSQL databases are non-relational",
                    "MongoDB uses BSON format"
            );

            // Pooling choice (explicit)
            PoolingType poolingType = PoolingType.AVG; // LEAF does not include pooler layer; AVG is typical

            // Single embedding test
            System.out.println("=== Single Embedding Sanity Check ===");
            float[] emb1 = safeEmbed(model, query1, poolingType);
            if (emb1 == null) {
                System.err.println("Embedding returned null for query: " + query1);
                System.exit(5);
            }
            if (l2Normalize) VectorMathUtils.l2normalize(emb1);
            printEmbeddingStats(emb1);

            // Verify shape
            if (emb1.length != embeddingLength) {
                System.err.printf(Locale.ROOT, "ERROR: Embedding length mismatch: expected %d got %d%n", embeddingLength, emb1.length);
                System.exit(6);
            }

            // Similarity checks
            System.out.println("=== Semantic Similarity Check ===");
            float[] emb2 = safeEmbed(model, query2, poolingType);
            float[] emb3 = safeEmbed(model, query3, poolingType);
            if (emb2 == null || emb3 == null) {
                System.err.println("ERROR: One of the test embeddings was null.");
                System.exit(7);
            }
            if (l2Normalize) {
                VectorMathUtils.l2normalize(emb2);
                VectorMathUtils.l2normalize(emb3);
            }
            float sim12 = VectorMathUtils.cosineSimilarity(emb1, emb2);
            float sim13 = VectorMathUtils.cosineSimilarity(emb1, emb3);
            System.out.println("Similarity (Query1, Query2): " + String.format("%.4f", sim12));
            System.out.println("Similarity (Query1, Query3): " + String.format("%.4f", sim13));
            if (sim12 > sim13) {
                System.out.println("✓ Related queries have higher similarity (as expected)");
            } else {
                System.out.println("⚠ WARNING: Related queries do NOT have higher similarity than unrelated one");
            }
            System.out.println();

            // Small IR-like retrieval test
            System.out.println("=== Small Information Retrieval Test ===");
            float[] base = safeEmbed(model, "MongoDB is a NoSQL database", poolingType);
            if (l2Normalize) VectorMathUtils.l2normalize(base);

            double maxSim = Double.NEGATIVE_INFINITY;
            int bestIndex = -1;
            for (int i = 0; i < batchInputs.size(); i++) {
                float[] e = safeEmbed(model, batchInputs.get(i), poolingType);
                if (l2Normalize) VectorMathUtils.l2normalize(e);
                double s = VectorMathUtils.cosineSimilarity(base, e);
                System.out.println(String.format("  [%d] Sim=%.4f - %s", i, s, batchInputs.get(i)));
                if (s > maxSim) {
                    maxSim = s;
                    bestIndex = i;
                }
            }
            System.out.println();
            System.out.println("Best match index: " + bestIndex + " (expected 0, 3 or 4). Score: " + String.format("%.4f", maxSim));
            if (bestIndex == 0 || bestIndex == 3 || bestIndex == 4) {
                System.out.println("Best match is MongoDB-related (as expected)");
            } else {
                System.out.println("! WARNING: Best match is not MongoDB-related");
            }
            System.out.println();

            // Batch embedding performance test
            System.out.println("=== Performance / Throughput Test ===");
            // Prepare repeated batch inputs to reach `iterations`
            List<String> perfInputs = new ArrayList<>(batchSize * iterations);
            for (int i = 0; i < iterations; i++) {
                for (int b = 0; b < batchSize; b++) {
                    // use the same short query for timing, but batch-based
                    perfInputs.add(query1 + " " + i + "-" + b);
                }
            }

            long t0 = System.nanoTime();
            float[][] results;
            if (batchSize <= 1) {
                // single-embedding mode repeated
                results = embedSequential(model, perfInputs, poolingType, l2Normalize, parallel);
            } else {
                // process in logical batches: for each iteration, embed batchSize items
                results = embedInBatches(model, perfInputs, batchSize, poolingType, l2Normalize, parallel);
            }
            long t1 = System.nanoTime();

            long elapsedNanos = t1 - t0;
            double elapsedMillis = elapsedNanos / 1_000_000.0;
            int totalEmbeddings = perfInputs.size();
            double avgMs = elapsedMillis / totalEmbeddings;
            System.out.println(String.format(Locale.ROOT, "Generated %d embeddings in %.2f ms (avg %.3f ms / embedding)",
                    totalEmbeddings, elapsedMillis, avgMs));
            System.out.println();

            System.out.println("=== Test Complete ===");
            model.close();
            System.exit(0);
        } catch (Exception ex) {
            System.err.println("Unexpected error during execution: " + ex.getMessage());
            ex.printStackTrace(System.err);
            System.exit(99);
        }
    }

    // Simple usage message
    private static void printUsageAndExit(int code) {
        System.out.println("Usage: LeafModelExample [--iterations N] [--batch-size N] [--normalize true|false] [--parallel true|false]");
        System.out.println("Defaults: iterations=100 batch-size=1 normalize=false parallel=false");
        System.exit(code);
    }

    // Safe wrapper for model.embed which handles exceptions and returns null on failure
    private static float[] safeEmbed(AbstractModel model, String text, PoolingType poolingType) {
        try {
            return model.embed(text, poolingType);
        } catch (Throwable t) {
            System.err.println("Error while embedding text: \"" + text + "\" -> " + t.getMessage());
            t.printStackTrace(System.err);
            return null;
        }
    }

    // Sequential embedding for a list of inputs; optional parallelization via parallel flag
    private static float[][] embedSequential(AbstractModel model, List<String> inputs, PoolingType poolingType, boolean normalize, boolean parallel) {
        int n = inputs.size();
        float[][] out = new float[n][];
        if (parallel) {
            // Simple parallel run using parallel streams — useful if the model implementation is thread-safe
            // WARNING: Only enable this if you're confident the model supports concurrent inference
            IntStream.range(0, n).parallel().forEach(i -> {
                float[] emb = safeEmbed(model, inputs.get(i), poolingType);
                if (emb != null && normalize) VectorMathUtils.l2normalize(emb);
                out[i] = emb;
            });
        } else {
            for (int i = 0; i < n; i++) {
                float[] emb = safeEmbed(model, inputs.get(i), poolingType);
                if (emb != null && normalize) VectorMathUtils.l2normalize(emb);
                out[i] = emb;
            }
        }
        return out;
    }

    // Process inputs in logical batches (simple loop), returns flattened results
    private static float[][] embedInBatches(AbstractModel model, List<String> inputs, int batchSize, PoolingType poolingType, boolean normalize, boolean parallel) {
        int total = inputs.size();
        List<float[]> collected = new ArrayList<>(total);
        for (int i = 0; i < total; i += batchSize) {
            int end = Math.min(i + batchSize, total);
            List<String> slice = inputs.subList(i, end);
            // Here we call sequential embedding for the slice. If the model supports a native batch API, replace this.
            float[][] batchResults = embedSequential(model, slice, poolingType, normalize, parallel);
            for (float[] r : batchResults) collected.add(r);
        }
        return collected.toArray(new float[0][]);
    }

    // Print basic stats on embedding vector
    private static void printEmbeddingStats(float[] emb) {
        boolean hasNonZero = false;
        boolean allFinite = true;
        float min = Float.MAX_VALUE;
        float max = -Float.MAX_VALUE;
        double sum = 0.0;
        if (emb == null || emb.length == 0) {
            System.out.println("Embedding is empty or null.");
            return;
        }
        for (float v : emb) {
            if (v != 0.0f) hasNonZero = true;
            if (!Float.isFinite(v)) allFinite = false;
            if (v < min) min = v;
            if (v > max) max = v;
            sum += v;
        }
        double mean = sum / emb.length;
        System.out.println("Embedding length: " + emb.length);
        System.out.println("Has non-zero values: " + hasNonZero);
        System.out.println("All values finite: " + allFinite);
        System.out.println(String.format(Locale.ROOT, "Value range: [%.6f, %.6f]", min, max));
        System.out.println(String.format(Locale.ROOT, "Mean value: %.6f", mean));
    }
}
