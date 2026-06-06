package io.teknek.deliverance.model.tensorparallel;

public record TensorParallelShardPlan(
        ShardRange queryHeads,
        ShardRange keyValueHeads,
        ShardRange attentionColumns,
        ShardRange keyValueColumns,
        ShardRange mlpIntermediate
) {
}
