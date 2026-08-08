package io.teknek.deliverance.safetensors;

import io.teknek.deliverance.tensor.AbstractTensor;

/**
 * One resolved, already-dtype-converted, already-scale-pre-multiplied LoRA delta for a single
 * base tensor (e.g. one layer's {@code q_proj.weight}), ready to feed directly into the two
 * matmuls that compute {@code scale * ((x @ loraA^T) @ loraB^T)} with no further per-call
 * conversion work -- see step 4 plan Section 1 item 2.
 *
 * <p>{@code loraA} has shape {@code [rank, inFeatures]}; {@code scaledLoraB} has shape
 * {@code [outFeatures, rank]} and already has the adapter's {@code alpha/r} scale baked in, so
 * applying this delta never needs a separate {@code TensorOperations#scale} call.</p>
 */
public record LoraLayerDelta(AbstractTensor loraA, AbstractTensor scaledLoraB, int rank) {
}
