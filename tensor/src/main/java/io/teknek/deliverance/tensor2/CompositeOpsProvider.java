package io.teknek.deliverance.tensor2;

import io.teknek.dysfx.Either;

/** Optional provider accelerators for semantic composite operations. */
public interface CompositeOpsProvider {
    default boolean supportsActivationMultiplyQuantize(ActivationMultiplyQuantize operation) {
        return false;
    }

    default Either<OpSupport, Void> activationMultiplyQuantize(ActivationMultiplyQuantize operation) {
        return Either.Left(OpSupport.Unsupported);
    }
}
