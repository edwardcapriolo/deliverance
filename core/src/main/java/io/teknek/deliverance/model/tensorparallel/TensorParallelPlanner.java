package io.teknek.deliverance.model.tensorparallel;

import io.teknek.deliverance.safetensors.Config;

/**
 * Computes static tensor-parallel shard ownership for a model config and rank context.
 *
 * <p>The planner does not perform communication and does not load weights. It only answers whether a model can be split
 * evenly for a requested tensor-parallel size and, when it can, which contiguous ranges belong to the current rank.</p>
 */
public final class TensorParallelPlanner {
    private TensorParallelPlanner() {
    }

    /**
     * Returns whether the model dimensions divide evenly across {@code size} tensor-parallel ranks.
     *
     * <p>Compatibility means the basic attention, KV, and MLP dimensions can be split without remainders. Rounding down
     * to a compatible size belongs in assignment selection, not in this method.</p>
     */
    public static boolean compatible(Config config, int size) {
        if (size < 1) {
            return false;
        }
        return config.numberOfHeads % size == 0
                && config.numberOfKeyValueHeads % size == 0
                && config.attentionLength % size == 0
                && config.kvLength % size == 0
                && config.hiddenLength % size == 0;
    }

    /**
     * Computes the shard plan for the supplied rank.
     *
     * <p>The returned ranges are half-open, local ownership ranges over global model dimensions. For example,
     * {@code attentionColumns=[512,1024)} means this rank owns that global slice of attention projection output/input
     * dimensions, depending on the specific weight policy.</p>
     */
    public static TensorParallelShardPlan plan(Config config, TensorParallelContext context) {
        requireCompatible(config, context.size());
        return new TensorParallelShardPlan(
                range(config.numberOfHeads, context),
                range(config.numberOfKeyValueHeads, context),
                range(config.attentionLength, context),
                range(config.kvLength, context),
                range(config.hiddenLength, context)
        );
    }

    /**
     * Throws if {@code context.size()} is not compatible with {@code config}.
     */
    public static void validate(Config config, TensorParallelContext context) {
        requireCompatible(config, context.size());
    }

    /**
     * Chooses the largest compatible tensor-parallel size no greater than {@code availableNodes}.
     *
     * <p>If 10 nodes are available but the model only divides cleanly by 4, this returns 4 and leaves the remaining nodes
     * idle for this static assignment.</p>
     */
    public static int chooseSize(Config config, int availableNodes) {
        if (availableNodes < 1) {
            throw new IllegalArgumentException("availableNodes must be >= 1");
        }
        for (int size = availableNodes; size >= 1; size--) {
            if (compatible(config, size)) {
                return size;
            }
        }
        return 1;
    }

    /**
     * Computes the half-open range owned by {@code context.rank()} for a dimension of length {@code total}.
     *
     * <p>{@code total} must divide evenly by {@code context.size()}. The returned range is in global coordinates; later
     * shard loading converts it into local dense tensors.</p>
     */
    public static ShardRange range(int total, TensorParallelContext context) {
        if (total < 0) {
            throw new IllegalArgumentException("total must be >= 0");
        }
        if (total % context.size() != 0) {
            throw new IllegalArgumentException("total " + total + " is not divisible by tensor parallel size "
                    + context.size());
        }
        int length = total / context.size();
        int start = context.rank() * length;
        return new ShardRange(start, start + length);
    }

    private static void requireCompatible(Config config, int size) {
        if (!compatible(config, size)) {
            throw new IllegalArgumentException("config is not compatible with tensor parallel size " + size);
        }
    }
}
