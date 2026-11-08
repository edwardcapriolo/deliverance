package io.teknek.reckoning.safetensors;

import io.teknek.reckoning.DType;
import io.teknek.reckoning.tensor2.Lighter;
import io.teknek.reckoning.tensor2.Scale;
import io.teknek.reckoning.tensor2.TensorRef;

/**
 * Dtype-conversion helpers shared by {@link MergingWeightLoader} (Phase 1, merge-at-load) and
 * {@link ResolvedLoraAdapter} (Phase 2, runtime hot-swap) for converting a {@code LoraAdapter}'s
 * stored-dtype {@code loraA}/{@code loraB} tensors to whatever dtype the caller needs to combine
 * them with.
 *
 * <p>Extracted from {@code MergingWeightLoader}'s originally-private helpers of the same shape --
 * see step 3 plan Section 11 for why that PR kept them private, and step 4 plan Section 2 for why
 * this PR (the second real caller) is the point to extract.</p>
 */
final class LoraTensorMath {

    private LoraTensorMath() {
    }

    static TensorRef toDType(TensorRef src, DType target, Lighter lighter) {
        validateDenseDType(target);
        if (src.dType() == target) {
            return src;
        }
        return lighter.reshape(src, target);
    }

    static TensorRef scaledCopy(TensorRef src, DType target, float factor, Lighter lighter) {
        validateDenseDType(target);
        TensorRef converted = lighter.reshape(src, target);
        if (factor == 1.0f) {
            return converted;
        }
        try {
            lighter.scale(new Scale(factor).target(converted).offsetAndLength(0,
                    Math.toIntExact(converted.shape().last())));
        } catch (RuntimeException e) {
            converted.close();
            throw e;
        }
        return converted;
    }

    private static void validateDenseDType(DType dType) {
        if (dType != DType.F32 && dType != DType.BF16 && dType != DType.F16) {
            throw new UnsupportedOperationException("Unsupported dtype for LoRA tensor math: " + dType);
        }
    }
}
