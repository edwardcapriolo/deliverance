package io.teknek.deliverance.safetensors;

import com.codahale.metrics.MetricRegistry;
import com.codahale.metrics.Timer;
import io.teknek.deliverance.safetensors.fetch.LoraAdapterModelFetcher;
import org.junit.jupiter.api.Test;

import java.util.Locale;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Establishes a profiling pattern for LoRA adapter loading, in the same spirit as {@code
 * core.model.InferenceProfiler} and the existing benchmark harness ({@code
 * core/benchmarking.md}). This module cannot depend on {@code core.model.InferenceProfiler}
 * directly -- {@code core} depends on {@code safetensors}, not the reverse, so importing it
 * here would create a circular module dependency -- so this instruments with the same
 * underlying Dropwizard {@link MetricRegistry}/{@link Timer} primitives {@code
 * InferenceProfiler} itself wraps, using the same dotted-lowercase naming convention
 * ({@code loraadapter.fetch}, {@code loraadapter.parse_header}, {@code
 * loraadapter.load_tensor}) so the numbers read the same way once this does get wired into
 * model loading in a later PR.
 *
 * <p>This test only has adapter fetch/parse/tensor-materialization phases to time -- there's
 * no merge or forward-pass cost yet, since nothing in {@code core} references this code. The
 * fuller benchmark comparing base-vs-adapted inference throughput belongs to Phase 1
 * (the next PR), once there's an actual adapted model to run prompts through.</p>
 */
public class LoraAdapterProfileTest {

    @Test
    void profilesAdapterFetchAndLoadPhases() {
        MetricRegistry metricRegistry = new MetricRegistry();
        LoraAdapterModelFetcher fetcher = new LoraAdapterModelFetcher("bunnycore", "Llama-3.2-1b-chatml-lora_model");

        try (LoraAdapter adapter = LoraAdapter.fromPretrained(fetcher, metricRegistry)) {
            adapter.deltaFor("model.layers.0.self_attn.q_proj.weight");
            adapter.deltaFor("model.layers.1.self_attn.q_proj.weight");
            adapter.deltaFor("model.layers.2.mlp.down_proj.weight");
        }

        printSummary(metricRegistry);

        assertTrue(metricRegistry.timer("loraadapter.fetch").getCount() > 0, "expected at least one fetch timing");
        assertTrue(metricRegistry.timer("loraadapter.parse_header").getCount() > 0, "expected at least one parse_header timing");
        assertTrue(metricRegistry.timer("loraadapter.load_tensor").getCount() > 0, "expected at least one load_tensor timing");
    }

    private static void printSummary(MetricRegistry metricRegistry) {
        System.out.println("[profile] LoraAdapter");
        for (Map.Entry<String, Timer> entry : metricRegistry.getTimers().entrySet()) {
            Timer timer = entry.getValue();
            double meanMs = timer.getSnapshot().getMean() / 1_000_000.0;
            double totalMs = timer.getCount() * meanMs;
            System.out.printf(Locale.ROOT, "[profile] %-30s count=%6d total_ms=%10.3f mean_ms=%10.3f%n",
                    entry.getKey(), timer.getCount(), totalMs, meanMs);
        }
    }
}
