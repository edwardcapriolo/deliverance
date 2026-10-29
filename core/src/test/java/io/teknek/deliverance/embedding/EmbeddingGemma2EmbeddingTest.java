package io.teknek.deliverance.embedding;

import io.dropwizard.metrics5.MetricName;
import io.dropwizard.metrics5.MetricRegistry;
import io.dropwizard.metrics5.Timer;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForEmbeddings;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.tensor2.TensorProviderKind;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.util.Comparator;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Simple real-checkpoint example for calculating an EmbeddingGemma 2 text embedding. */
class EmbeddingGemma2EmbeddingTest {
    @Test
    @Tag("large-model")
    void calculatesA768DimensionalTextEmbedding() {
        String text = "This is a test document about machine learning";
        ModelFetcher fetcher = new ModelFetcher("google", "embeddinggemma-2");
        File modelDirectory = fetcher.maybeDownload();
        AbstractModel model = AutoModelForEmbeddings.newBuilder(fetcher).buildLocalEmbeddingModel();
        try (model) {
            model.getLighter().removeTensorOperations(TensorProviderKind.NAIVE);
            boolean previousProfilerState = InferenceProfiler.isEnabled();
            for (int k = 0; k < 5; k++) {
                InferenceProfiler.setEnabled(true);
                try {
                    InferenceProfiler.reset();
                    long embeddingStart = System.nanoTime();
                    float[] embedding = model.embed(text, PoolingType.AVG);
                    double embeddingMillis = (System.nanoTime() - embeddingStart) / 1_000_000.0;

                    assertEquals(768, embedding.length, "EmbeddingGemma 2 native output should have 768 dimensions");
                    for (float value : embedding) {
                        assertTrue(Float.isFinite(value), "Embedding values must be finite");
                    }
                    InferenceProfiler.printSummary("embeddinggemma2.one_shot", 100);
                    System.out.printf(java.util.Locale.ROOT,
                            "EmbeddingGemma 2 total embedding time: %.3f ms%n", embeddingMillis);
                    System.out.println("EmbeddingGemma 2 embedding dimensions: " + embedding.length);
                    System.out.println("EmbeddingGemma 2 first values: " + embedding[0] + ", " + embedding[1]
                            + ", " + embedding[2] + ", " + embedding[3]);
                    if (k == 4) {
                        printRegisteredTimerMetrics(model.getMetricRegistry());
                    }
                } finally {
                    InferenceProfiler.setEnabled(previousProfilerState);
                }
            }
        }
    }

    private static void printRegisteredTimerMetrics(MetricRegistry metrics) {
        System.out.println("EmbeddingGemma 2 registered timer metrics:");
        metrics.getTimers().entrySet().stream()
                .sorted(Comparator.comparingDouble((Map.Entry<MetricName, Timer> entry) ->
                        entry.getValue().getCount() * entry.getValue().getSnapshot().getMean()).reversed())
                .forEach(entry -> {
                    Timer timer = entry.getValue();
                    double totalMillis = timer.getCount() * timer.getSnapshot().getMean() / 1_000_000.0;
                    System.out.printf(java.util.Locale.ROOT,
                            "[metric] %s count=%d total_ms=%.3f mean_us=%.3f%n",
                            entry.getKey(), timer.getCount(), totalMillis,
                            timer.getSnapshot().getMean() / 1_000.0);
                });
    }
}
