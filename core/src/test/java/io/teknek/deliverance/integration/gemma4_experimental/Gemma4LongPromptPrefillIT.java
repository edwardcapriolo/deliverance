package io.teknek.deliverance.integration.gemma4_experimental;

import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor2.TensorRef;
import io.teknek.deliverance.tensor.kv.KvCacheSession;
import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.LinkedHashMap;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertTrue;

/** Characterizes long-prompt batch prefill against token-by-token prefill without generation sampling. */
class Gemma4LongPromptPrefillIT {
    @Test
    void longPromptBatchPrefillMatchesTokenByTokenPrefill() {
        AbstractModel model = Gemma4Suite.getOrCreate();
        PromptContext prompt = model.promptSupport().orElseThrow().builder()
                .addTemplateArgs(Map.of("enable_thinking", false))
                .addUserMessage("""
                        Take your time and evaluate multiple alternative explanations.
                        You have to choose between buying gold or buying silver. What do you buy? Explain why in detail.
                        Use historical facts and recent market trends to justify.
                        """)
                .build();
        int[] tokens = model.constructPromptTokensForRuntime(prompt.getPrompt());
        Map<String, float[]> batchTrace = new LinkedHashMap<>();
        Map<String, float[]> stepTrace = new LinkedHashMap<>();

        model.setLayerDebugHook(event -> {
            if ("layer_scalar".equals(event.stage()) || event.layerIndex() <= 1) {
                batchTrace.put(event.layerIndex() + ":" + event.stage(), values(event.hiddenStates()));
            }
        });
        TensorRef batch;
        try (KvCacheSession batchSession = model.newKvCacheSession()) {
            batch = model.batchForwardRef(tokens, 0, batchSession);
        } finally {
            model.clearLayerDebugHook();
        }
        try (batch; KvCacheSession stepSession = model.newKvCacheSession()) {
            model.setLayerDebugHook(event -> {
                if ("layer_scalar".equals(event.stage()) || event.layerIndex() <= 1) {
                    stepTrace.put(event.layerIndex() + ":" + event.stage(), values(event.hiddenStates()));
                }
            });
            TensorRef step = null;
            try {
                for (int position = 0; position < tokens.length; position++) {
                    if (step != null) step.close();
                    step = model.forwardRef(tokens[position], position, stepSession, Optional.empty());
                }
                Drift drift = lastRowDrift(batch, step);
                System.out.printf("gemma4_long_prompt_prefill_drift max_abs=%.8f mean_abs=%.8f%n",
                        drift.maxAbs(), drift.meanAbs());
                printLayerDrift(batchTrace, stepTrace);
                assertTrue(drift.maxAbs() < 1.0e-2f,
                        "Long prompt batch prefill diverged from token-by-token prefill: " + drift);
            } finally {
                if (step != null) step.close();
                model.clearLayerDebugHook();
            }
        }
    }

    private static float[] values(AbstractTensor tensor) {
        float[] values = new float[Math.toIntExact(tensor.size())];
        int index = 0;
        for (int row = 0; row < tensor.shape().first(); row++) {
            for (int column = 0; column < tensor.shape().last(); column++) values[index++] = tensor.get(row, column);
        }
        return values;
    }

    private static void printLayerDrift(Map<String, float[]> batchTrace, Map<String, float[]> stepTrace) {
        for (Map.Entry<String, float[]> entry : batchTrace.entrySet()) {
            float[] other = stepTrace.get(entry.getKey());
            if (other == null) continue;
            float max = 0.0f;
            for (int i = 0; i < other.length; i++) max = Math.max(max, Math.abs(entry.getValue()[i] - other[i]));
            System.out.printf("gemma4_long_prompt_stage_drift stage=%s max_abs=%.8f%n", entry.getKey(), max);
        }
    }

    private static Drift lastRowDrift(TensorRef batch, TensorRef step) {
        int batchRow = batch.shape().first() - 1;
        float max = 0.0f;
        double total = 0.0;
        for (int column = 0; column < batch.shape().last(); column++) {
            float difference = Math.abs(batch.get(batchRow, column) - step.get(0, column));
            max = Math.max(max, difference);
            total += difference;
        }
        return new Drift(max, total / batch.shape().last());
    }

    private record Drift(float maxAbs, double meanAbs) {
        @Override
        public String toString() {
            return "max_abs=" + maxAbs + ", mean_abs=" + meanAbs;
        }
    }
}
