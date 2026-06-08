package io.teknek.deliverance.model.tensorparallel;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class DefaultTransformerWeightPolicyResolverTest {

    private final DefaultTransformerWeightPolicyResolver resolver = new DefaultTransformerWeightPolicyResolver();

    @Test
    public void commonAttentionInputProjectionsAreColumnParallel() {
        assertEquals(TensorParallelWeightPolicy.QUERY_PROJECTION,
                resolver.resolve("model.layers.0.self_attn.q_proj.weight"));
        assertEquals(TensorParallelWeightPolicy.KEY_VALUE_PROJECTION,
                resolver.resolve("model.layers.0.self_attn.k_proj.weight"));
        assertEquals(TensorParallelWeightPolicy.KEY_VALUE_PROJECTION,
                resolver.resolve("model.layers.0.self_attn.v_proj.weight"));
    }

    @Test
    public void commonAttentionOutputProjectionIsRowParallel() {
        assertEquals(TensorParallelWeightPolicy.ATTENTION_OUTPUT_PROJECTION,
                resolver.resolve("model.layers.0.self_attn.o_proj.weight"));
    }

    @Test
    public void commonMlpProjectionsUseExpectedPolicies() {
        assertEquals(TensorParallelWeightPolicy.MLP_INPUT_PROJECTION,
                resolver.resolve("model.layers.0.mlp.gate_proj.weight"));
        assertEquals(TensorParallelWeightPolicy.MLP_INPUT_PROJECTION,
                resolver.resolve("model.layers.0.mlp.up_proj.weight"));
        assertEquals(TensorParallelWeightPolicy.MLP_OUTPUT_PROJECTION,
                resolver.resolve("model.layers.0.mlp.down_proj.weight"));
    }

    @Test
    public void normsEmbeddingsAndUnknownWeightsAreReplicated() {
        assertEquals(TensorParallelWeightPolicy.REPLICATED,
                resolver.resolve("model.layers.0.input_layernorm.weight"));
        assertEquals(TensorParallelWeightPolicy.REPLICATED,
                resolver.resolve("model.embed_tokens.weight"));
        assertEquals(TensorParallelWeightPolicy.REPLICATED,
                resolver.resolve("lm_head.weight"));
        assertEquals(TensorParallelWeightPolicy.REPLICATED,
                resolver.resolve("model.layers.0.self_attn.q_proj.bias"));
    }
}
