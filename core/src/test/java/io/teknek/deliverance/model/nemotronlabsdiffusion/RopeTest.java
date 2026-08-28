package io.teknek.deliverance.model.nemotronlabsdiffusion;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Ported from upstream Hugging Face `tests/utils/test_modeling_rope_utils.py::RopeTest`. */
class RopeTest {
    @Test
    void test_yarn_rope_numerically() {
        float[] expectedInvFreq = new float[] {
                1.0000e+00f, 8.6596e-01f, 7.4989e-01f, 6.4938e-01f, 5.6234e-01f, 4.8697e-01f,
                4.2170e-01f, 3.6517e-01f, 3.1623e-01f, 2.7384e-01f, 2.3714e-01f, 2.0535e-01f,
                1.7783e-01f, 1.5399e-01f, 1.3335e-01f, 1.1548e-01f, 1.0000e-01f, 8.3479e-02f,
                6.9590e-02f, 5.7925e-02f, 4.8136e-02f, 3.9931e-02f, 3.3061e-02f, 2.7315e-02f,
                2.2515e-02f, 1.8512e-02f, 1.5177e-02f, 1.2403e-02f, 1.0101e-02f, 8.1924e-03f,
                6.6143e-03f, 5.3120e-03f, 4.2400e-03f, 3.3599e-03f, 2.6396e-03f, 2.0520e-03f,
                1.5746e-03f, 1.1882e-03f, 8.7713e-04f, 6.2810e-04f, 4.3007e-04f, 2.7384e-04f,
                2.3714e-04f, 2.0535e-04f, 1.7783e-04f, 1.5399e-04f, 1.3335e-04f, 1.1548e-04f,
                1.0000e-04f, 8.6596e-05f, 7.4989e-05f, 6.4938e-05f, 5.6234e-05f, 4.8697e-05f,
                4.2170e-05f, 3.6517e-05f, 3.1623e-05f, 2.7384e-05f, 2.3714e-05f, 2.0535e-05f,
                1.7783e-05f, 1.5399e-05f, 1.3335e-05f, 1.1548e-05f
        };

        NemotronLabsDiffusionRope.YarnParameters defaults = NemotronLabsDiffusionRope.computeYarnParameters(
                128, 4096, 32, 2048,
                Map.of("rope_type", "yarn", "rope_theta", 10_000.0d, "factor", 10.0d));
        assertEquals((float) (0.1d * Math.log(10.0d) + 1.0d), defaults.attentionScaling(), 1.0e-6f);

        NemotronLabsDiffusionRope.YarnParameters explicitAttention = NemotronLabsDiffusionRope.computeYarnParameters(
                128, 4096, 32, 2048,
                Map.of("rope_type", "yarn", "rope_theta", 10_000.0d, "factor", 10.0d,
                        "attention_factor", 0.5d));
        assertEquals(0.5f, explicitAttention.attentionScaling(), 0.0f);

        float[] defaultInvFreq = defaultInvFreq(128, 10_000.0d);
        NemotronLabsDiffusionRope.YarnParameters bounded = NemotronLabsDiffusionRope.computeYarnParameters(
                128, 4096, 32, 2048,
                Map.of("rope_type", "yarn", "rope_theta", 10_000.0d, "factor", 10.0d,
                        "beta_fast", 32, "beta_slow", 1));
        for (int i = 0; i < bounded.invFreq().length; i++) {
            assertTrue(bounded.invFreq()[i] >= defaultInvFreq[i] / 10.0f - 1.0e-8f, "lower bound i=" + i);
            assertTrue(bounded.invFreq()[i] <= defaultInvFreq[i] + 1.0e-8f, "upper bound i=" + i);
        }

        NemotronLabsDiffusionRope.YarnParameters highBetaFast = NemotronLabsDiffusionRope.computeYarnParameters(
                128, 4096, 32, 2048,
                Map.of("rope_type", "yarn", "rope_theta", 10_000.0d, "factor", 10.0d,
                        "beta_fast", 1000, "beta_slow", 1));
        assertFalse(highBetaFast.invFreq()[0] < defaultInvFreq[0] + 1.0e-8f);
        for (int i = 1; i < highBetaFast.invFreq().length; i++) {
            assertTrue(highBetaFast.invFreq()[i] < defaultInvFreq[i] + 1.0e-8f, "interpolating i=" + i);
        }
        for (int i = highBetaFast.invFreq().length - 20; i < highBetaFast.invFreq().length; i++) {
            assertEquals(defaultInvFreq[i] / 10.0f, highBetaFast.invFreq()[i], 1.0e-7f, "tail i=" + i);
        }

        for (int i = 0; i < expectedInvFreq.length; i++) {
            assertEquals(expectedInvFreq[i], bounded.invFreq()[i], 5.0e-6f, "snapshot i=" + i);
        }
    }

    @Test
    void test_get_llama_4_attn_scale() {
        NemotronLabsDiffusionRope rope = new NemotronLabsDiffusionRope(128, 4096, 32, 4096,
                Map.of("rope_type", "yarn", "rope_theta", 1_000_000.0d, "factor", 0.25d,
                        "original_max_position_embeddings", 16_384, "llama_4_scaling_beta", 0.1d));

        assertEquals(1.0f, rope.llama4QueryScale(0), 0.0f);
        assertEquals(1.0f, rope.llama4QueryScale(16_383), 0.0f);
        assertEquals((float) (1.0d + 0.1d * Math.log(2.0d)), rope.llama4QueryScale(16_384), 1.0e-6f);
        assertEquals((float) (1.0d + 0.1d * Math.log(3.0d)), rope.llama4QueryScale(32_768), 1.0e-6f);
    }

    private static float[] defaultInvFreq(int headDim, double theta) {
        float[] invFreq = new float[headDim / 2];
        for (int i = 0; i < invFreq.length; i++) {
            invFreq[i] = (float) (1.0d / Math.pow(theta, (2.0d * i) / headDim));
        }
        return invFreq;
    }
}
