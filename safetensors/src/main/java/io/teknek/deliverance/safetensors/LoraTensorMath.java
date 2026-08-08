package io.teknek.deliverance.safetensors;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.impl.BFloat16BufferTensor;
import io.teknek.deliverance.tensor.impl.Float16BufferTensor;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;

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

    static AbstractTensor toDType(AbstractTensor src, DType target) {
        if (src.dType() == target) {
            return src;
        }
        return scaledCopy(src, target, 1.0f);
    }

    static AbstractTensor scaledCopy(AbstractTensor src, DType target, float factor) {
        AbstractTensor converted = allocateLike(target, src.shape().first(), src.shape().last());
        for (int row = 0; row < src.shape().first(); row++) {
            for (int col = 0; col < src.shape().last(); col++) {
                converted.set(src.get(row, col) * factor, row, col);
            }
        }
        return converted;
    }

    static AbstractTensor allocateLike(DType dType, int rows, int cols) {
        TensorShape shape = TensorShape.of(rows, cols);
        return switch (dType) {
            case F32 -> new FloatBufferTensor(shape);
            case BF16 -> new BFloat16BufferTensor(shape);
            case F16 -> new Float16BufferTensor(shape);
            default -> throw new UnsupportedOperationException("Unsupported dtype for LoRA tensor math: " + dType);
        };
    }
}
