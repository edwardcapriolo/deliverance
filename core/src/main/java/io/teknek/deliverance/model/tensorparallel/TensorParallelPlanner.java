package io.teknek.deliverance.model.tensorparallel;

import io.teknek.deliverance.safetensors.Config;

public final class TensorParallelPlanner {
    private TensorParallelPlanner() {
    }

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

    public static void validate(Config config, TensorParallelContext context) {
        requireCompatible(config, context.size());
    }

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
