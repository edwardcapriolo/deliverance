package io.teknek.deliverance.generator;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.NaiveTensorOperations;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.when;

public class Gemma4CausalSelfAttentionTest {
    @Test
    public void scoreAppliesHeadDimensionScaling() {
        try (AttentionFixture fixture = new AttentionFixture(null);
             AbstractTensor query = queryTensor();
             AbstractTensor key = keyTensor()) {
            float score = fixture.attention.score(query, 0, 0, key, 0, 0);
            assertEquals(35.0f, score, 1.0e-5f);
        }
    }

    @Test
    public void scoreAppliesSoftCapAfterScaling() {
        try (AttentionFixture fixture = new AttentionFixture(10.0f);
             AbstractTensor query = queryTensor();
             AbstractTensor key = keyTensor()) {
            float score = fixture.attention.score(query, 0, 0, key, 0, 0);
            assertEquals((float) (Math.tanh(3.5d) * 10.0d), score, 1.0e-5f);
        }
    }

    @Test
    public void softmaxNormalizesScoresAndPreservesOrdering() {
        try (AttentionFixture fixture = new AttentionFixture(null)) {
            float[] scores = new float[]{1.0f, 2.0f, 3.0f};
            fixture.attention.softmax(scores);
            assertEquals(1.0f, scores[0] + scores[1] + scores[2], 1.0e-5f);
            assertTrue(scores[2] > scores[1]);
            assertTrue(scores[1] > scores[0]);
        }
    }

    @Test
    public void fillVisibleRowsPacksWindowAcrossPages() {
        try (AttentionFixture fixture = new AttentionFixture(null);
             AbstractTensor page1 = matrix(2, 4, 0);
             AbstractTensor page2 = matrix(3, 4, 8);
             AbstractTensor packed = new FloatBufferTensor(4, 4)) {
            int rows = fixture.attention.fillVisibleRows(packed, new AbstractTensor[]{page1, page2}, 4, 1, 4);
            assertEquals(4, rows);
            assertEquals(4.0f, packed.get(0, 0), 1.0e-5f);
            assertEquals(7.0f, packed.get(0, 3), 1.0e-5f);
            assertEquals(8.0f, packed.get(1, 0), 1.0e-5f);
            assertEquals(11.0f, packed.get(1, 3), 1.0e-5f);
            assertEquals(12.0f, packed.get(2, 0), 1.0e-5f);
            assertEquals(15.0f, packed.get(2, 3), 1.0e-5f);
            assertEquals(16.0f, packed.get(3, 0), 1.0e-5f);
            assertEquals(19.0f, packed.get(3, 3), 1.0e-5f);
        }
    }

    @Test
    public void fillVisibleRowsFromDenseUsesContiguousWindow() {
        try (AttentionFixture fixture = new AttentionFixture(null);
             AbstractTensor dense = matrix(5, 4, 0);
             AbstractTensor packed = new FloatBufferTensor(3, 4)) {
            int rows = fixture.attention.fillVisibleRowsFromDense(packed, dense, 1, 3, 4);
            assertEquals(3, rows);
            assertEquals(4.0f, packed.get(0, 0), 1.0e-5f);
            assertEquals(7.0f, packed.get(0, 3), 1.0e-5f);
            assertEquals(8.0f, packed.get(1, 0), 1.0e-5f);
            assertEquals(11.0f, packed.get(1, 3), 1.0e-5f);
            assertEquals(12.0f, packed.get(2, 0), 1.0e-5f);
            assertEquals(15.0f, packed.get(2, 3), 1.0e-5f);
        }
    }

    private static AbstractTensor queryTensor() {
        AbstractTensor query = new FloatBufferTensor(1, 4);
        query.set(1.0f, 0, 0);
        query.set(2.0f, 0, 1);
        query.set(3.0f, 0, 2);
        query.set(4.0f, 0, 3);
        return query;
    }

    private static AbstractTensor keyTensor() {
        AbstractTensor key = new FloatBufferTensor(1, 4);
        key.set(5.0f, 0, 0);
        key.set(6.0f, 0, 1);
        key.set(7.0f, 0, 2);
        key.set(8.0f, 0, 3);
        return key;
    }

    private static AbstractTensor matrix(int rows, int cols, int start) {
        FloatBufferTensor tensor = new FloatBufferTensor(rows, cols);
        int value = start;
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                tensor.set(value++, row, col);
            }
        }
        return tensor;
    }

    private static final class AttentionFixture implements AutoCloseable {
        private final AbstractTensor q = new FloatBufferTensor(8, 8);
        private final AbstractTensor qNorm = new FloatBufferTensor(8, 8);
        private final AbstractTensor k = new FloatBufferTensor(4, 8);
        private final AbstractTensor v = new FloatBufferTensor(4, 8);
        private final AbstractTensor kNorm = new FloatBufferTensor(4, 8);
        private final AbstractTensor o = new FloatBufferTensor(8, 8);
        private final Gemma4CausalSelfAttention attention;

        private AttentionFixture(Float attnLogitSoftCapping) {
            io.teknek.deliverance.model.AbstractModel model = Mockito.mock(io.teknek.deliverance.model.AbstractModel.class);
            Map<String, Object> entries = new LinkedHashMap<>();
            entries.put("max_position_embeddings", 16);
            entries.put("hidden_size", 8);
            entries.put("intermediate_size", 16);
            entries.put("num_attention_heads", 2);
            entries.put("num_key_value_heads", 1);
            entries.put("num_hidden_layers", 1);
            entries.put("rms_norm_eps", 1.0e-6);
            entries.put("vocab_size", 32);
            entries.put("bos_token_id", 2);
            entries.put("eos_token_id", List.of(1));
            entries.put("hidden_activation", "gelu_pytorch_tanh");
            entries.put("head_dim", 4);
            entries.put("sliding_window", 8);
            entries.put("layer_types", List.of("sliding_attention"));
            entries.put("rope_parameters", Map.of("sliding_attention", Map.of("rope_theta", 10000.0)));
            if (attnLogitSoftCapping != null) {
                entries.put("attn_logit_softcapping", attnLogitSoftCapping);
            }
            io.teknek.deliverance.model.gemma4.Gemma4Config config = new io.teknek.deliverance.model.gemma4.Gemma4Config(
                    entries,
                    List.of("Gemma4ForConditionalGeneration"),
                    List.of(1)
            );
            when(model.getConfig()).thenReturn(config);

            ConfigurableTensorProvider provider = new ConfigurableTensorProvider(new NaiveTensorOperations());
            this.attention = new Gemma4CausalSelfAttention(
                    model,
                    0,
                    "sliding_attention",
                    false,
                    false,
                    q,
                    qNorm,
                    Optional.of(k),
                    Optional.of(v),
                    Optional.of(kNorm),
                    o,
                    provider,
                    new MetricRegistry()
            );
        }

        @Override
        public void close() {
            q.close();
            qNorm.close();
            k.close();
            v.close();
            kNorm.close();
            o.close();
        }
    }
}
