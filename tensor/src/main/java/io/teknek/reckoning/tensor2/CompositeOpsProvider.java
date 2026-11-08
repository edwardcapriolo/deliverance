package io.teknek.reckoning.tensor2;

import io.teknek.dysfx.Either;

/** Optional provider accelerators for semantic composite operations. */
public interface CompositeOpsProvider {
    default boolean supportsGemmAccumulate(GemmAccumulate operation) {
        return false;
    }

    default Either<OpSupport, Void> gemmAccumulate(GemmAccumulate operation) {
        return Either.Left(OpSupport.Unsupported);
    }

    default boolean supportsActivationMultiplyQuantize(ActivationMultiplyQuantize operation) {
        return false;
    }

    default Either<OpSupport, Void> activationMultiplyQuantize(ActivationMultiplyQuantize operation) {
        return Either.Left(OpSupport.Unsupported);
    }

    default boolean supportsDecodePagedAttention(DecodePagedAttention operation) {
        return false;
    }

    default Either<OpSupport, Void> decodePagedAttention(DecodePagedAttention operation) {
        return Either.Left(OpSupport.Unsupported);
    }
}
