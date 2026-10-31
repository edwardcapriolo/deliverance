package io.teknek.deliverance.generator2;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.gemma4.Gemma4Config;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;
import io.teknek.deliverance.tensor2.CompositeOps;
import io.teknek.deliverance.tensor.kv.CacheExecutionMode;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Characterizes the Gemma4 implementation that runs on the KV cache v2 path. */
class Gemma4CausalSelfAttention2Test {
    @Test
    void usesUnitAttentionScaleAndGemmaQkvShapes() throws Exception {
        try (Fixture fixture = new Fixture();
             TensorRef query = fixture.tensor(1, 8);
             TensorRef key = fixture.tensor(1, 4);
             TensorRef value = fixture.tensor(1, 4)) {
            Field scale = KvCacheSelfAttention2.class.getDeclaredField("attentionScale");
            scale.setAccessible(true);
            assertEquals(1.0f, scale.get(fixture.attention));
            assertEquals(8, query.shape().last());
            assertEquals(4, key.shape().last());
            assertEquals(4, value.shape().last());
            fill(query, 1.0f);
            for (int column = 4; column < 8; column++) query.set(2.0f, 0, column);
            fill(key, 2.0f);
            fill(value, 3.0f);
            invoke(fixture.attention, "normalizeQueryKey", query, key);
            invoke(fixture.attention, "normalizeValue", value);
            assertEquals(1.0f, query.get(0, 0), 1.0e-5f);
            assertEquals(1.0f, query.get(0, 3), 1.0e-5f);
            assertEquals(1.0f, query.get(0, 4), 1.0e-5f);
            assertEquals(1.0f, query.get(0, 7), 1.0e-5f);
            assertEquals(1.0f, key.get(0, 0), 1.0e-5f);
            assertEquals(1.0f, value.get(0, 0), 1.0e-5f);
        }
    }

    @Test
    void retainsV2WindowPackingAndSoftmaxSemantics() throws Exception {
        try (Fixture fixture = new Fixture();
             TensorRef page1 = matrix(2, 4, 0);
             TensorRef page2 = matrix(3, 4, 8);
             TensorRef packed = fixture.tensor(4, 4);
             TensorRef scores = fixture.tensor(1, 3)) {
            int rows = (int) invoke(fixture.attention, "fillVisibleRows", packed,
                    new TensorRef[]{page1, page2}, 4, 1, 4);
            assertEquals(4, rows);
            assertEquals(4.0f, packed.get(0, 0));
            assertEquals(8.0f, packed.get(1, 0));
            assertEquals(16.0f, packed.get(3, 0));
            scores.set(1.0f, 0, 0);
            scores.set(2.0f, 0, 1);
            scores.set(3.0f, 0, 2);
            invoke(fixture.attention, "softmax", scores, 3);
            assertEquals(1.0f, scores.get(0, 0) + scores.get(0, 1) + scores.get(0, 2), 1.0e-5f);
            assertTrue(scores.get(0, 2) > scores.get(0, 1));
            assertTrue(scores.get(0, 1) > scores.get(0, 0));
        }
    }

    @Test
    void bidirectionalGemma4DisablesCausalWithinBlockMask() throws Exception {
        try (Fixture fixture = new Fixture(true)) {
            assertEquals(false, invoke(fixture.attention, "causalWithinBlock",
                    CacheExecutionMode.PREFILL_UPDATE_CACHE));
        }
    }

    @Test
    void compactRopeMatchesReferenceRotationIncludingProportionalIdentityTail() {
        Lighter lighter = new Lighter(new MetricRegistry());
        try (TensorRef target = lighter.allocate(DType.F32, TensorShape.of(2, 4))) {
            fill(target, 1.0f);
            float[][] frequencies = new float[8][2];
            for (int position = 0; position < 4; position++) {
                double angle = position * 0.25;
                frequencies[position * 2] = new float[] {(float) Math.cos(angle), (float) Math.sin(angle)};
                frequencies[position * 2 + 1] = new float[] {1.0f, 0.0f};
            }
            new CompositeOps(lighter).rotaryEmbedding(target, 1, 4, 2, frequencies);
            for (int row = 0; row < 2; row++) {
                double angle = (row + 2) * 0.25;
                assertEquals(Math.cos(angle) - Math.sin(angle), target.get(row, 0), 1.0e-5f);
                assertEquals(Math.sin(angle) + Math.cos(angle), target.get(row, 2), 1.0e-5f);
                assertEquals(1.0f, target.get(row, 1), 1.0e-5f);
                assertEquals(1.0f, target.get(row, 3), 1.0e-5f);
            }
        }
    }

    private static Object invoke(Object target, String name, Object... args) throws Exception {
        for (Class<?> type = target.getClass(); type != null; type = type.getSuperclass()) {
            for (Method method : type.getDeclaredMethods()) {
                if (method.getName().equals(name) && method.getParameterCount() == args.length) {
                    method.setAccessible(true);
                    return method.invoke(target, args);
                }
            }
        }
        throw new NoSuchMethodException(name);
    }

    private static void fill(TensorRef tensor, float value) {
        if (tensor.dims() == 1) {
            for (int index = 0; index < tensor.shape().first(); index++) tensor.set(value, index);
            return;
        }
        for (int row = 0; row < tensor.shape().first(); row++)
            for (int column = 0; column < tensor.shape().last(); column++) tensor.set(value, row, column);
    }

    private static TensorRef matrix(int rows, int columns, int start) {
        Lighter lighter = new Lighter();
        TensorRef tensor = lighter.allocate(DType.F32, TensorShape.of(rows, columns));
        int value = start;
        for (int row = 0; row < rows; row++)
            for (int column = 0; column < columns; column++) tensor.set(value++, row, column);
        return tensor;
    }

    private static final class Fixture implements AutoCloseable {
        private final Lighter lighter = new Lighter(new MetricRegistry());
        private final Gemma4CausalSelfAttention2 attention;

        private Fixture() { this(false); }

        private Fixture(boolean bidirectional) {
            Gemma4Config config = new Gemma4Config(new LinkedHashMap<>(Map.ofEntries(
                    Map.entry("max_position_embeddings", 16), Map.entry("hidden_size", 8),
                    Map.entry("intermediate_size", 16), Map.entry("num_attention_heads", 2),
                    Map.entry("num_key_value_heads", 1), Map.entry("num_hidden_layers", 2),
                    Map.entry("rms_norm_eps", 1.0e-6), Map.entry("vocab_size", 32),
                    Map.entry("bos_token_id", 2), Map.entry("eos_token_id", List.of(1)),
                    Map.entry("hidden_activation", "gelu_pytorch_tanh"), Map.entry("head_dim", 4),
                    Map.entry("layer_types", List.of("sliding_attention", "full_attention")),
                    Map.entry("sliding_window", 4),
                    Map.entry("rope_parameters", Map.of(
                            "sliding_attention", Map.of("rope_theta", 10000.0),
                            "full_attention", Map.of("rope_theta", 1000000.0)))
            )), List.of("Gemma4ForConditionalGeneration"), List.of(1));
            AbstractModel model = Mockito.mock(AbstractModel.class);
            Mockito.when(model.getConfig()).thenReturn(config);
            TensorRef queryNorm = ones(8);
            for (int column = 4; column < 8; column++) queryNorm.set(2.0f, 0, column);
            attention = new Gemma4CausalSelfAttention2(model, 0,
                    tensor(8, 8), queryNorm, tensor(4, 8), tensor(4, 8), ones(4), tensor(8, 8),
                    lighter, new MetricRegistry(), 2, 1, 4, 8, 4,
                    config.ropeInvFreqsByLayerType.get("sliding_attention"), "q", "k", "v", "o",
                    false, bidirectional, 0, 0);
        }

        private TensorRef tensor(int... shape) { return lighter.allocate(DType.F32, TensorShape.of(shape)); }

        private TensorRef ones(int... shape) {
            TensorRef tensor = tensor(shape);
            fill(tensor, 1.0f);
            return tensor;
        }

        @Override
        public void close() { }
    }
}
