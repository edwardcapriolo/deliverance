package io.teknek.deliverance.model.gemma4;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.CompositeOps;
import io.teknek.deliverance.tensor2.MultiplyAccumulate;
import io.teknek.deliverance.tensor2.TensorRef;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.List;
import java.util.Map;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;

class Gemma4MoeFeedForward2Test {
    @Test
    void packedExpertsAreFiniteDeterministicAndWeightSensitive() {
        int hidden = 4;
        int intermediate = 3;
        Gemma4Config config = config(hidden, intermediate);
        MetricRegistry metrics = new MetricRegistry();
        Lighter lighter = new Lighter(metrics);
        AbstractModel model = Mockito.mock(AbstractModel.class);
        Mockito.when(model.getConfig()).thenReturn(config);
        Mockito.when(model.getMetricRegistry()).thenReturn(metrics);
        Mockito.when(model.getWorkingQType()).thenReturn(DType.F32);
        Mockito.when(model.makeDenseTensorRef(Mockito.anyInt(), Mockito.anyInt()))
                .thenAnswer(invocation -> lighter.allocate(DType.F32,
                        TensorShape.of(invocation.getArgument(0), invocation.getArgument(1))));
        Mockito.when(model.makeDenseTensorRef(Mockito.any(TensorShape.class)))
                .thenAnswer(invocation -> lighter.allocate(DType.F32, invocation.getArgument(0)));

        try (TensorRef input = lighter.allocate(DType.F32, TensorShape.of(1, hidden));
             TensorRef router = lighter.allocate(DType.F32, TensorShape.of(2, hidden));
             TensorRef routerScale = lighter.allocate(DType.F32, TensorShape.of(1, hidden));
             TensorRef expertScale = lighter.allocate(DType.F32, TensorShape.of(1, 2));
             TensorRef gateUp = lighter.allocate(DType.F32, TensorShape.of(2, intermediate * 2, hidden));
             TensorRef down = lighter.allocate(DType.F32, TensorShape.of(2, hidden, intermediate))) {
            for (int i = 0; i < hidden; i++) {
                input.set(1.0f, 0, i);
                routerScale.set(1.0f, 0, i);
            }
            expertScale.set(1.0f, 0, 0);
            expertScale.set(1.0f, 0, 1);
            for (int i = 0; i < hidden; i++) {
                router.set(100.0f, 0, i);
                router.set(-100.0f, 1, i);
            }
            for (int expert = 0; expert < 2; expert++) {
                for (int row = 0; row < intermediate * 2; row++) {
                    for (int column = 0; column < hidden; column++) {
                        gateUp.set(expert == 0 ? 0.25f : 0.5f, expert, row, column);
                    }
                }
                for (int row = 0; row < hidden; row++) {
                    for (int column = 0; column < intermediate; column++) {
                        down.set(0.25f, expert, row, column);
                    }
                }
            }
            Gemma4MoeFeedForward2 block = new Gemma4MoeFeedForward2(model, config, router, routerScale,
                    expertScale, gateUp, down, lighter);
            assertArrayEquals(new int[] { intermediate * 2, hidden }, gateUp.slice(0).shape().shapeArray());
            assertArrayEquals(new int[] { hidden, intermediate }, down.slice(0).shape().shapeArray());
            try (TensorRef source = lighter.allocate(DType.F32, TensorShape.of(1, hidden));
                 TensorRef gate = lighter.allocate(DType.F32, TensorShape.of(1, intermediate));
                 TensorRef up = lighter.allocate(DType.F32, TensorShape.of(1, intermediate))) {
                lighter.copy(input, 0, source, 0, hidden);
                lighter.dotProductRows(gate, source, gateUp.slice(0), 0, 0, hidden, 0, intermediate, 0);
                lighter.dotProductRows(up, source, gateUp.slice(0), 0, 0, hidden, intermediate, intermediate, 0);
                assertNotEquals(0.0f, gate.get(0, 0));
                assertNotEquals(0.0f, up.get(0, 0));
                new CompositeOps(lighter).silu(gate);
                lighter.multiplyAccumulate(new MultiplyAccumulate(up).into(gate)
                        .offsetAndLength(0, intermediate));
                try (TensorRef downOutput = lighter.allocate(DType.F32, TensorShape.of(1, hidden))) {
                    assertNotEquals(0.0f, gate.get(0, 0));
                    lighter.dotProductRows(downOutput, gate, down.slice(0), 0, 0, intermediate, 0, hidden, 0);
                    assertNotEquals(0.0f, downOutput.get(0, 0));
                }
            }
            float[] baseline = new float[hidden];
            try (TensorRef result = block.forward(input, Optional.empty())) {
                for (int i = 0; i < hidden; i++) {
                    baseline[i] = result.get(0, i);
                    assertTrue(Float.isFinite(baseline[i]));
                }
            }
            try (TensorRef repeat = block.forward(input, Optional.empty())) {
                for (int i = 0; i < hidden; i++) assertEquals(baseline[i], repeat.get(0, i), 1.0e-6f);
            }
            float[] expertChanged = new float[hidden];
            for (int expert = 0; expert < 2; expert++) {
                for (int row = 0; row < intermediate * 2; row++) {
                    for (int column = 0; column < hidden; column++) {
                        gateUp.set(expert == 0 ? 1.5f : 2.0f, expert, row, column);
                    }
                }
            }
            try (TensorRef changedExpert = block.forward(input, Optional.empty())) {
                for (int i = 0; i < hidden; i++) {
                    expertChanged[i] = changedExpert.get(0, i);
                }
                assertTrue(maxDrift(baseline, changedExpert) > 1.0e-5f);
            }
            for (int i = 0; i < hidden; i++) {
                router.set(-100.0f, 0, i);
                router.set(100.0f, 1, i);
            }
            try (TensorRef changedRouter = block.forward(input, Optional.empty())) {
                assertTrue(maxDrift(expertChanged, changedRouter) > 1.0e-5f);
                for (int i = 0; i < hidden; i++) assertTrue(Float.isFinite(changedRouter.get(0, i)));
            }
        }
    }

    private static float maxDrift(float[] expected, TensorRef actual) {
        float max = 0.0f;
        for (int i = 0; i < expected.length; i++) {
            max = Math.max(max, Math.abs(expected[i] - actual.get(0, i)));
        }
        return max;
    }

    private static Gemma4Config config(int hidden, int intermediate) {
        Map<String, Object> text = Map.ofEntries(
                Map.entry("max_position_embeddings", 32), Map.entry("hidden_size", hidden),
                Map.entry("intermediate_size", intermediate), Map.entry("num_attention_heads", 1),
                Map.entry("num_key_value_heads", 1), Map.entry("num_hidden_layers", 1),
                Map.entry("rms_norm_eps", 1.0e-6), Map.entry("vocab_size", 32),
                Map.entry("bos_token_id", 0), Map.entry("eos_token_id", List.of(1)),
                Map.entry("hidden_activation", "silu"), Map.entry("head_dim", 4),
                Map.entry("layer_types", List.of("full_attention")),
                Map.entry("rope_parameters", Map.of("full_attention", Map.of("rope_type", "default"))),
                Map.entry("enable_moe_block", true), Map.entry("num_experts", 2),
                Map.entry("top_k_experts", 1), Map.entry("moe_intermediate_size", intermediate));
        return new Gemma4Config(text, List.of(), List.of(1));
    }
}
