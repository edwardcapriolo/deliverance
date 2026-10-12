package io.teknek.deliverance.model.embeddinggemma2;

import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;
import io.teknek.deliverance.tensor.TensorShape;

import java.util.Objects;

/**
 * EmbeddingGemma 2 token lookup followed by embedding scaling.
 *
 * <p>The reference operation is {@code embed_tokens(input_ids) * sqrt(hidden_size)}.
 * The row gather and multiply are delegated to {@link Lighter}, keeping model
 * orchestration on the TensorRef path and leaving the provider boundary
 * available for a fused gather kernel.</p>
 */
public final class EmbeddingGemma2ScaledWordEmbeddingOperation {
    /**
     * Gathers token rows and multiplies every output element by {@code scale}.
     *
     * @param output writable TensorRef shaped {@code [inputIds.length, hidden]}
     * @param embeddingTable vocabulary TensorRef shaped {@code [vocabulary, hidden]}
     * @param inputIds token IDs to gather
     * @param scale embedding scale, normally {@code sqrt(hiddenSize)}
     * @param lighter TensorRef operation/provider owner
     */
    public void apply(TensorRef output, TensorRef embeddingTable, int[] inputIds, float scale, Lighter lighter) {
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(embeddingTable, "embeddingTable");
        Objects.requireNonNull(inputIds, "inputIds");
        Objects.requireNonNull(lighter, "lighter");
        if (!Float.isFinite(scale)) {
            throw new IllegalArgumentException("scale must be finite");
        }
        int hidden = (int) embeddingTable.shape().last();
        if (embeddingTable.dims() != 2 || output.dims() != 2
                || output.shape().first() != inputIds.length || output.shape().last() != hidden) {
            throw new IllegalArgumentException("embedding output and table must have compatible 2D shapes");
        }
        if (output.dType() == embeddingTable.dType()) {
            lighter.gatherRowsScale(output, embeddingTable, inputIds, scale);
            return;
        }
        try (TensorRef gathered = lighter.allocate(embeddingTable.dType(),
                TensorShape.of(inputIds.length, hidden))) {
            lighter.gatherRowsScale(gathered, embeddingTable, inputIds, scale);
            lighter.reshape(gathered, output);
        }
    }
}
