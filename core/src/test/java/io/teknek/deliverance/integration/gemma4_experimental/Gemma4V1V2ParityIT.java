package io.teknek.deliverance.integration.gemma4_experimental;

import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.KvBufferCache;
import io.teknek.deliverance.tensor.kv.KvCacheSession;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.NaiveTensorOperations;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Arrays;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.assertTrue;

/** Short cached-checkpoint parity trace for the two Gemma4 implementations. */
public class Gemma4V1V2ParityIT {
    private static final String LEGACY_PROPERTY = "deliverance.gemma4.legacy";

    @AfterEach
    void clearLegacySwitch() {
        System.clearProperty(LEGACY_PROPERTY);
    }

    @Test
    void tracesFirstV1V2Divergence() {
        ModelFetcher fetch = new ModelFetcher("google", "gemma-4-E2B-it");
        Assumptions.assumeTrue(fetch.pathForModel().toFile().isDirectory());
        int[] tokens;
        AbstractModel v1 = null;
        AbstractModel v2 = null;
        try {
            tokens = buildTokens(fetch);
            System.setProperty(LEGACY_PROPERTY, "true");
            v1 = AutoModelForCausaLm.newBuilder(fetch)
                    .withTensorProvider(new ConfigurableTensorProvider(new NaiveTensorOperations()))
                    .buildLocalTransformerModel();
            Trace legacy = traceLegacy(v1, tokens);
            v1.close();
            v1 = null;
            System.setProperty(LEGACY_PROPERTY, "false");
            v2 = AutoModelForCausaLm.newBuilder(fetch).buildLocalTransformerModel();
            Trace modern = traceV2(v2, tokens);
            String first = null;
            for (Map.Entry<String, float[]> entry : legacy.values.entrySet()) {
                float[] other = modern.values.get(entry.getKey());
                if (other == null) continue;
                float drift = maxDrift(entry.getValue(), other);
                System.out.printf("gemma4_v1_v2 stage=%s drift=%.8f%n", entry.getKey(), drift);
                if (first == null && drift > 1.0e-3f) first = entry.getKey() + " drift=" + drift;
            }
            System.out.println("gemma4_v1_v2 first_divergence=" + first);
            assertTrue(first == null || first.contains("query_") || first.contains("key_")
                            || first.contains("value_") || first.startsWith("layer_output")
                            || first.startsWith("ple_input"),
                    "unexpected parity trace stage: " + first);
        } finally {
            if (v1 != null) v1.close();
            if (v2 != null) v2.close();
        }
    }

    private static int[] buildTokens(ModelFetcher fetch) {
        System.setProperty(LEGACY_PROPERTY, "true");
        try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetch).buildLocalTransformerModel()) {
            return Arrays.stream(model.encodeForRuntime("Answer in one short sentence: what is 2 plus 2?"))
                    .mapToInt(Math::toIntExact).toArray();
        }
    }

    private static Trace traceLegacy(AbstractModel model, int[] tokens) {
        Trace trace = new Trace();
        model.setLayerDebugHook(trace.consumer());
        try (KvBufferCache.KvBuffer cache = model.newKvBuffer();
             AbstractTensor ignored = model.batchForward(tokens, 0, cache)) {
        }
        return trace;
    }

    private static Trace traceV2(AbstractModel model, int[] tokens) {
        Trace trace = new Trace();
        model.setLayerDebugHook(trace.consumer());
        try (KvCacheSession session = model.newKvCacheSession();
             io.teknek.deliverance.tensor2.TensorRef ignored = model.batchForwardRef(tokens, 0, session)) {
            assertTrue(session.isHeterogeneous(), "E2B V2 cache must retain heterogeneous layer widths");
        }
        return trace;
    }

    private static float maxDrift(float[] left, float[] right) {
        float result = 0.0f;
        for (int i = 0; i < left.length; i++) result = Math.max(result, Math.abs(left[i] - right[i]));
        return result;
    }

    private static final class Trace {
        private final Map<String, float[]> values = new LinkedHashMap<>();

        Consumer<AbstractModel.LayerDebugEvent> consumer() {
            return event -> {
                AbstractTensor tensor = event.hiddenStates();
                float[] values = new float[Math.toIntExact(tensor.size())];
                int index = 0;
                for (int row = 0; row < tensor.shape().first(); row++) {
                    for (int column = 0; column < tensor.shape().last(); column++) {
                        values[index++] = tensor.get(row, column);
                    }
                }
                this.values.put(event.layerIndex() + ":" + event.stage(), values);
            };
        }
    }
}
