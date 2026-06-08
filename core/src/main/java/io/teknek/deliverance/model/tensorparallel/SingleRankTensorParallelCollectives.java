package io.teknek.deliverance.model.tensorparallel;

import io.teknek.deliverance.tensor.AbstractTensor;

/**
 * Tensor-parallel collectives implementation for a group containing exactly one rank.
 *
 * <p>For a single-rank group, an all-reduce sum is the identity operation: the sum of the only contribution is that
 * contribution. This implementation returns {@code local} directly. Callers must close the returned tensor exactly once.</p>
 */
public class SingleRankTensorParallelCollectives implements TensorParallelCollectives {
    @Override
    public AbstractTensor allReduceSum(String key, AbstractTensor local) {
        return local;
    }
}
