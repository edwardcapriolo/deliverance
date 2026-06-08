package io.teknek.deliverance.model.tensorparallel;

import io.teknek.deliverance.tensor.AbstractTensor;

/**
 * Collective communication operations used by tensor-parallel model execution.
 *
 * <p>A collective is an operation that all tensor-parallel ranks in the same group must enter with compatible inputs.
 * Implementations may communicate across JVMs, processes, devices, or may be local test implementations. The interface
 * deliberately does not define membership discovery or rank assignment; those are handled by {@link TensorParallelContext}
 * and the deployment layer.</p>
 */
public interface TensorParallelCollectives {

    /**
     * Sums one tensor contribution from every tensor-parallel rank and returns the same reduced tensor value to each rank.
     *
     * <p>This is the tensor-parallel operation needed after row-parallel projections. Each rank computes a partial output
     * over its local weight shard. The all-reduce sum combines those partial outputs into the full output tensor.</p>
     *
     * <p>All ranks participating in the same tensor-parallel group must call this method with the same {@code key} for
     * the same logical operation. The {@code key} identifies the synchronization point, for example
     * {@code "layer.3.mlp.down_proj"} or {@code "layer.3.self_attn.o_proj"}. Implementations may use the key to match
     * peer contributions and detect mismatched or out-of-order collective calls.</p>
     *
     * <p>The {@code local} tensor is this rank's partial contribution. All ranks must provide tensors with compatible
     * shape and dtype. The returned tensor must contain the element-wise sum of every rank's local tensor:</p>
     *
     * <pre>
     * result = local_rank_0 + local_rank_1 + ... + local_rank_N
     * </pre>
     *
     * <p>Ownership rule: the caller retains ownership of {@code local}. The returned tensor is owned by the caller and
     * must be closed by the caller when no longer needed. Implementations may return {@code local} itself only when no
     * communication/reduction is required, such as tensor-parallel size 1.</p>
     *
     * @param key stable logical operation key shared by all ranks for this collective call
     * @param local this rank's local partial tensor contribution
     * @return tensor containing the element-wise sum across all ranks
     */
    AbstractTensor allReduceSum(String key, AbstractTensor local);
}