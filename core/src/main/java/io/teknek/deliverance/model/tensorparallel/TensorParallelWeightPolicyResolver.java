package io.teknek.deliverance.model.tensorparallel;

/**
 * Resolves tensor-parallel sharding policy for model weights.
 *
 * <p>New model implementations should provide or validate a resolver instead of relying blindly on suffix defaults.
 * Incorrect policies can produce shape-correct but numerically wrong models.</p>
 */
public interface TensorParallelWeightPolicyResolver {
    TensorParallelWeightPolicy resolve(String weightName);
}
