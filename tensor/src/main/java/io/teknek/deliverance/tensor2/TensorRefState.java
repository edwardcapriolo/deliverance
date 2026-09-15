package io.teknek.deliverance.tensor2;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;

import java.util.Map;
import java.util.Objects;

record TensorRefState(
        LeaseState leaseState,
        Allocator allocator,
        Tensor underlying,
        TensorShape shape,
        DType dType,
        int stride,
        String device,
        Map<String, TensorRef> sidecars,
        TensorRefState previous
) {
    TensorRefState {
        Objects.requireNonNull(leaseState, "leaseState");
        sidecars = sidecars == null ? Map.of() : Map.copyOf(sidecars);
    }

    static TensorRefState closed(TensorRefState previous) {
        return new TensorRefState(LeaseState.UNUSED, null, null, null, null, 0, null, Map.of(), previous);
    }
}
