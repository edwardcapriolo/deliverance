package io.teknek.deliverance.generator;

import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.safetensors.LoraLayerDelta;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.operations.TensorOperations;

/**
 * Applies one resolved LoRA delta to a projection's output: {@code output += scale * ((input @
 * loraA^T) @ loraB^T)}, computed as two small matmuls plus an accumulate -- see step 4 plan
 * Section 1 item 2.
 *
 * <p>{@code input} is whatever activation tensor the base projection itself already consumes at
 * that hook site (e.g. {@code input}/{@code lnemb} for q/k/v/gate/up, {@code vq}/{@code bufq} for
 * o_proj/down_proj) -- callers do not need to pre-convert it. Every real hook site's activation
 * arrives already quantized to {@code model.getWorkingQType()} (either because
 * {@code TransformerBlock} quantizes the block's hidden state before calling
 * attention/feed-forward, or because the projection quantizes its own local input just before the
 * base matmul), which the underlying SIMD/native matmul kernels only pair with a narrow, specific
 * set of weight-side dtypes (e.g. a native {@code BF16} activation only pairs with a {@code BF16}
 * or {@code Q4} weight-side operand, never dense {@code F32}) -- confirmed the hard way, via
 * {@code LoraHotSwapIntegrationTest} throwing {@code UnsupportedOperationException: BF16 F32}
 * against a real model. See step 4 plan Section 11 item 9 (revised) for the full story, including
 * why an earlier design that fed a hand-picked *dense* tensor into each hook site instead was
 * wrong in the opposite direction.</p>
 *
 * <p>To keep {@code loraA}/{@code scaledLoraB} at one precision-preserving dense dtype
 * ({@code ResolvedLoraAdapter} resolves them to {@code model.getWorkingDType()}) regardless of
 * what dtype the activation happens to arrive in, this dequantizes {@code input} to match
 * {@code loraA}'s dtype first -- a no-op copy-avoidance when it's already a match (the common case
 * once {@code workingQType == workingDType}, e.g. an explicit dense-working-memory config).</p>
 *
 * <p>Called once per targeted projection per {@code forward()} call, unchunked -- matching how the
 * existing bias-accumulate calls this mirrors are themselves unchunked, not the projection's own
 * chunked {@code VectorMath.pchunk} convention.</p>
 */
final class LoraDeltaApplier {

    private LoraDeltaApplier() {
    }

    static void apply(AbstractModel model, AbstractTensor output, AbstractTensor input, LoraLayerDelta delta) {
        int outFeatures = output.shape().last();
        TensorOperations tensorOps = model.primaryTensorOperations();
        AbstractTensor denseInput = toDType(model, input, delta.loraA().dType());
        int batchSize = denseInput.shape().first();
        int inFeatures = denseInput.shape().last();
        try (AbstractTensor rankResult = model.getTensorAllocator()
                .getDirty(model.getWorkingDType(), TensorShape.of(batchSize, delta.rank()));
                AbstractTensor deltaResult = model.getTensorAllocator()
                        .getDirty(model.getWorkingDType(), TensorShape.of(batchSize, outFeatures))) {
            tensorOps.dotProductChunk(rankResult, denseInput, delta.loraA(), 0, inFeatures, 0, delta.rank());
            tensorOps.dotProductChunk(deltaResult, rankResult, delta.scaledLoraB(), 0, delta.rank(), 0, outFeatures);
            tensorOps.accumulate(output, deltaResult, 0, outFeatures);
        } finally {
            if (denseInput != input) {
                denseInput.close();
            }
        }
    }

    /**
     * Returns {@code t} unchanged if it's already {@code target}, otherwise a freshly pooled
     * {@code target}-dtype copy -- generic element-by-element conversion via {@code get}/{@code
     * set} (every {@code AbstractTensor} dequantizes on read regardless of its own backing dtype),
     * the same technique {@code LoraTensorMath}/{@code MergingWeightLoader} already use to convert
     * an adapter's on-disk dtype to a base model's dtype. Callers must close the result only when
     * it isn't {@code t} itself -- see the {@code finally} block in {@link #apply}.
     */
    private static AbstractTensor toDType(AbstractModel model, AbstractTensor t, io.teknek.deliverance.DType target) {
        if (t.dType() == target) {
            return t;
        }
        AbstractTensor converted = model.getTensorAllocator().getDirty(target, t.shape());
        int rows = t.shape().first();
        int cols = t.shape().last();
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                converted.set(t.get(row, col), row, col);
            }
        }
        return converted;
    }
}
