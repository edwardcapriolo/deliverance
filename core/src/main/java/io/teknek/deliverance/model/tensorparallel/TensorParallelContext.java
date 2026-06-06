package io.teknek.deliverance.model.tensorparallel;

/**
 * Static tensor-parallel identity for one model process.
 *
 * <p>A tensor-parallel group has {@link #size()} participants. Each participant has one zero-based {@link #rank()} in
 * that group. Rank and size determine which shard of model weights, attention heads, KV heads, and intermediate MLP
 * dimensions the process owns.</p>
 */
public interface TensorParallelContext {

    /**
     * Returns this process' zero-based position in the tensor-parallel group.
     *
     * <p>For a group of size 4, valid ranks are {@code 0}, {@code 1}, {@code 2}, and {@code 3}. The rank is stable for
     * the lifetime of the loaded model.</p>
     */
    int rank();

    /**
     * Returns the number of participants in the tensor-parallel group.
     *
     * <p>A size of {@code 1} means tensor parallelism is disabled and the single process owns the full model.</p>
     */
    int size();

    /**
     * Returns {@code true} when the model is split across more than one participant.
     */
    default boolean enabled() {
        return size() > 1;
    }

    /**
     * Returns {@code true} for rank 0, the rank expected to coordinate request-level generation work.
     */
    default boolean coordinatorRank() {
        return rank() == 0;
    }
}
