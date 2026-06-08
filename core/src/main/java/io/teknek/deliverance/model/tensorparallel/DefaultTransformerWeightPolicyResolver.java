package io.teknek.deliverance.model.tensorparallel;

/**
 * Default tensor-parallel policy resolver for common decoder-only transformer weight names.
 *
 * <p>This resolver covers the common Llama/Gemma/Qwen/Mistral-style projection suffixes. Architectures with packed QKV,
 * MoE, multimodal, or otherwise nonstandard tensors should override these defaults.</p>
 */
public class DefaultTransformerWeightPolicyResolver implements TensorParallelWeightPolicyResolver {
    @Override
    public TensorParallelWeightPolicy resolve(String weightName) {
        if (weightName.endsWith(".q_proj.weight")) {
            return TensorParallelWeightPolicy.QUERY_PROJECTION;
        }
        if (endsWithAny(weightName, ".k_proj.weight", ".v_proj.weight")) {
            return TensorParallelWeightPolicy.KEY_VALUE_PROJECTION;
        }
        if (weightName.endsWith(".o_proj.weight")) {
            return TensorParallelWeightPolicy.ATTENTION_OUTPUT_PROJECTION;
        }
        if (endsWithAny(weightName, ".gate_proj.weight", ".up_proj.weight")) {
            return TensorParallelWeightPolicy.MLP_INPUT_PROJECTION;
        }
        if (weightName.endsWith(".down_proj.weight")) {
            return TensorParallelWeightPolicy.MLP_OUTPUT_PROJECTION;
        }
        return TensorParallelWeightPolicy.REPLICATED;
    }

    private static boolean endsWithAny(String value, String... suffixes) {
        for (String suffix : suffixes) {
            if (value.endsWith(suffix)) {
                return true;
            }
        }
        return false;
    }
}
