package io.teknek.deliverance.model.tensorparallel;

/**
 * Describes how a model weight participates in tensor parallelism.
 *
 * <p>Most decoder-only transformer families follow the same pattern: Q/K/V and first MLP projections are
 * column-parallel, output/down projections are row-parallel, and norms/scalars are replicated. That default is useful,
 * but it is not universal. Model implementors must verify the policy for every new architecture, especially models with
 * packed QKV weights, MoE experts, tied heads, multimodal tensors, sidecar quantization tensors, or family-specific
 * projection names.</p>
 */
public enum TensorParallelWeightPolicy {
    /** The full weight is loaded on each rank. */
    REPLICATED,

    /** Query projection output features are split across ranks. */
    QUERY_PROJECTION,

    /** Key and value projection output features are split across ranks. */
    KEY_VALUE_PROJECTION,

    /** Attention output projection input features are split across ranks and require reducing partial outputs. */
    ATTENTION_OUTPUT_PROJECTION,

    /** MLP gate/up projection output features are split across ranks. */
    MLP_INPUT_PROJECTION,

    /** MLP down projection input features are split across ranks and require reducing partial outputs. */
    MLP_OUTPUT_PROJECTION
}
