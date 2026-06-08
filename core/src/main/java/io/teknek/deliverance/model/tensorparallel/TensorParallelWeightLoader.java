package io.teknek.deliverance.model.tensorparallel;

import io.teknek.deliverance.safetensors.TensorShardAxis;
import io.teknek.deliverance.safetensors.TensorShardSpec;
import io.teknek.deliverance.safetensors.WeightLoader;
import io.teknek.deliverance.tensor.AbstractTensor;

/**
 * Applies tensor-parallel weight sharding policy to a regular weight loader.
 *
 * <p>The policy names describe model math semantics. The actual storage-axis mapping assumes the common safetensors
 * projection layout {@code [output, input]}: projection-output sharding loads row shards, while projection-input
 * sharding loads column shards.</p>
 */
public class TensorParallelWeightLoader {
    private final WeightLoader delegate;
    private final TensorParallelContext context;
    private final TensorParallelShardPlan plan;
    private final TensorParallelWeightPolicyResolver resolver;

    public TensorParallelWeightLoader(WeightLoader delegate, TensorParallelContext context,
            TensorParallelShardPlan plan, TensorParallelWeightPolicyResolver resolver) {
        this.delegate = delegate;
        this.context = context;
        this.plan = plan;
        this.resolver = resolver;
    }

    public AbstractTensor load(String weightName) {
        if (!context.enabled()) {
            return delegate.load(weightName);
        }
        return switch (resolver.resolve(weightName)) {
            case REPLICATED -> delegate.load(weightName);
            case QUERY_PROJECTION -> delegate.load(weightName, rowShard(plan.attentionColumns()));
            case KEY_VALUE_PROJECTION -> delegate.load(weightName, rowShard(plan.keyValueColumns()));
            case ATTENTION_OUTPUT_PROJECTION -> delegate.load(weightName, columnShard(plan.attentionColumns()));
            case MLP_INPUT_PROJECTION -> delegate.load(weightName, rowShard(plan.mlpIntermediate()));
            case MLP_OUTPUT_PROJECTION -> delegate.load(weightName, columnShard(plan.mlpIntermediate()));
        };
    }

    private static TensorShardSpec rowShard(ShardRange range) {
        return new TensorShardSpec(TensorShardAxis.ROWS, range.startInclusive(), range.endExclusive());
    }

    private static TensorShardSpec columnShard(ShardRange range) {
        return new TensorShardSpec(TensorShardAxis.COLUMNS, range.startInclusive(), range.endExclusive());
    }
}
