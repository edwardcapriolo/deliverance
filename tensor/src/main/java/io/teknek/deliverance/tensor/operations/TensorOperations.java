package io.teknek.deliverance.tensor.operations;

import com.codahale.metrics.MetricRegistry;
import com.google.common.base.Preconditions;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.VectorTensorMathUtils;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;

public interface TensorOperations {

    String name();

    /**
     * This is the number of splits methods like pchunk will attempt to cut the input into. for GPU the value
     * is always 1, generally it is the size of the fork join pool backing the tensor operations but theoretically
     * it could be a different value
     * @return The number of splits to cut the dataset into for batch processing
     */
    int parallelSplitSize();

    /* This is the minimum the tensor provider supports. So for example if the model is Q4 and the provider falls below
    this the working memory will be set to at least this type
     */
    default DType preferredWorkingQuantizedType() {
        return DType.I8;
    }

    /**
     * Register a tensor with the operations provider.  This is used to optimize operations on the tensor (e.g. GPU Load).
     */
    default void registerModelTensor(AbstractTensor t) { }

    /** Creates an optional backend for providers that maintain device-side KV storage. */
    default io.teknek.deliverance.tensor.KvStorageBackend createKvStorageBackend() {
        return io.teknek.deliverance.tensor.KvStorageBackend.NONE;
    }

    default io.teknek.deliverance.tensor.KvStorageBackend createKvStorageBackend(MetricRegistry metricRegistry) {
        return createKvStorageBackend();
    }

    default float dotProduct(AbstractTensor a, AbstractTensor b, int limit) {
        return dotProduct(a, b, 0, 0, limit);
    }

    default float dotProduct(AbstractTensor a, AbstractTensor b, int aoffset, int boffset, int limit) {
        FloatBufferTensor r =  new FloatBufferTensor(TensorShape.ONE);
        batchDotProduct(r, a, b, aoffset, boffset, limit);
        return r.get(0, 0);
    }

    default void batchDotProduct(AbstractTensor result, AbstractTensor a, AbstractTensor b,
            int aColumnOffset, int bColumnOffset, int columnLimit) {
        batchDotProduct(result, a, b, aColumnOffset, bColumnOffset, columnLimit, 0, 0, b.shape().first());
    }

    void batchDotProduct(AbstractTensor result, AbstractTensor a, AbstractTensor b,
            int aColumnOffset, int bColumnOffset, int columnLimit, int rRowOffset, int bRowOffset, int rowChunkSize);

    default void dotProductChunk(AbstractTensor result, AbstractTensor a, AbstractTensor b,
            int columnOffset, int columnLimit, int rowOffset, int rowChunkSize) {
        batchDotProduct(result, a, b, columnOffset, columnOffset, columnLimit, 0, rowOffset, rowChunkSize);
    }

    default void dotProductBatchChunk(
            AbstractTensor[] result,
            AbstractTensor a,
            AbstractTensor[] b,
            int offset,
            int limit,
            int chunkStart,
            int chunkSize
    ) {
        Preconditions.checkArgument(b[0].dims() == 2 && result.length == b.length);
        for (int j = 0; j < result.length; j++) {
            dotProductChunk(result[j], a, b[j], offset, limit, chunkStart, chunkSize);
        }
    }

    /**
     * For each position in the tensor, add b into a.  Must be same size.
     */
    void accumulate(AbstractTensor a, AbstractTensor b, int offset, int length);

    /**
     * For each position in the tensor, multiply b into a.  Must be same size.
     */
    void maccumulate(AbstractTensor a, AbstractTensor b, int offset, int length);

    /**
     * Performs the BLAS SAXPY operation {@code y = alpha * x + y} over a contiguous vector window.
     *
     * <p>SAXPY means "single-precision A times X plus Y". This method updates {@code y} in place:</p>
     *
     * <pre>
     * for i in 0..limit:
     *     y[yoffset + i] += alpha * x[xoffset + i]
     * </pre>
     *
     * <p>The offsets are element offsets within the logical tensor row/vector being used. In attention this operation is
     * used to accumulate weighted value vectors into the current attention output.</p>
     */
    void saxpy(float alpha, AbstractTensor x, AbstractTensor y, int xoffset, int yoffset, int limit);

    /**
     * Performs repeated SAXPY operations using one scalar from {@code alpha} for each selected row of {@code x}.
     *
     * <p>This method updates {@code y} in place:</p>
     *
     * <pre>
     * int alphaIndex = aOffset;
     * for row in xRowOffset .. xRowOffset + batchSize:
     *     for i in 0..limit:
     *         y[yoffset + i] += alpha[alphaIndex] * x[row][xoffset + i]
     *     alphaIndex++
     * </pre>
     *
     * <p>In attention, {@code alpha} is usually the attention weight vector and {@code x} is a packed page/window of
     * value vectors. {@code aOffset} and {@code xRowOffset} allow callers to process only the portion of the attention
     * window that overlaps a specific KV page.</p>
     */
    default void saxpy(
            AbstractTensor alpha,
            AbstractTensor x,
            AbstractTensor y,
            int xoffset,
            int yoffset,
            int limit,
            int aOffset,
            int xRowOffset,
            int batchSize
    ) {
        Preconditions.checkArgument(y.shape().first() == 1);
        Preconditions.checkArgument(aOffset >= 0 && aOffset + batchSize <= alpha.shape().last());
        Preconditions.checkArgument(xRowOffset >= 0 && xRowOffset + batchSize <= x.shape().first());
        int batchLimit = xRowOffset + batchSize;
        for (int xi = xRowOffset; xi < batchLimit; xi++) {
            saxpy(alpha.get(0, aOffset++), x.slice(xi), y, xoffset, yoffset, limit);
        }
    }

    /**
     * For each position multiply value by the scale factor
     */
    void scale(float factor, AbstractTensor x, int offset, int length);

    /**
     * Quantizes the tensor to the specified type (if supported)
     */
    /*
    default AbstractTensor quantize(AbstractTensor t, DType qtype, int offset, int length) {
        AbstractTensor t2 = TensorCache.instance.get(t.dType(), t.shape());
        t2.copyFrom(t, offset, offset, length);
        return t2;
    }*/
    AbstractTensor quantize(AbstractTensor t, DType qtype, int offset, int length);

    /** Computes a dot product over two row slices. Providers may override with hardware-specific implementations. */
    default float dotSlice(AbstractTensor left, int leftRow, int leftOffset, AbstractTensor right, int rightRow,
            int rightOffset, int length) {
        float dot = 0.0f;
        for (int col = 0; col < length; col++) {
            dot += left.get(leftRow, leftOffset + col) * right.get(rightRow, rightOffset + col);
        }
        return dot;
    }

    default boolean usesOptimizedDotSlice(AbstractTensor left, AbstractTensor right) {
        return false;
    }

    /** Computes dot products between one row slice and a batch of row slices into {@code scores}. */
    default void dotRowsToArray(AbstractTensor left, int leftRow, int leftOffset, AbstractTensor rows,
            int rowOffset, int rowColumnOffset, int rowCount, int width, float[] scores, int scoresOffset) {
        for (int row = 0; row < rowCount; row++) {
            scores[scoresOffset + row] = dotSlice(left, leftRow, leftOffset, rows, rowOffset + row, rowColumnOffset,
                    width);
        }
    }

    default boolean usesOptimizedDotRowsToArray(AbstractTensor left, AbstractTensor rows) {
        return false;
    }

    /** Mutates {@code out = out * oldScale + value * weight} over one row slice. */
    default void weightedRescaleAccumulateSlice(AbstractTensor out, int outRow, int outOffset, AbstractTensor value,
            int valueRow, int valueOffset, int length, float oldScale, float weight) {
        for (int col = 0; col < length; col++) {
            out.set(out.get(outRow, outOffset + col) * oldScale + value.get(valueRow, valueOffset + col) * weight,
                    outRow, outOffset + col);
        }
    }

    default boolean usesOptimizedWeightedRescaleAccumulateSlice(AbstractTensor out, AbstractTensor value) {
        return false;
    }

    /** Mutates {@code out += value * weight} over one row slice. */
    default void accumulateWeightedSlice(AbstractTensor out, int outRow, int outOffset, AbstractTensor value,
            int valueRow, int valueOffset, int length, float weight) {
        for (int col = 0; col < length; col++) {
            out.set(out.get(outRow, outOffset + col) + value.get(valueRow, valueOffset + col) * weight,
                    outRow, outOffset + col);
        }
    }

    default boolean usesOptimizedAccumulateWeightedSlice(AbstractTensor out, AbstractTensor value) {
        return false;
    }

    /** Mutates {@code out += sum_i weights[i] * rows[rowOffset + i]} over one row slice. */
    default void accumulateWeightedRows(AbstractTensor out, int outRow, int outOffset, AbstractTensor rows,
            int rowOffset, int rowColumnOffset, int rowCount, int width, float[] weights, int weightsOffset) {
        for (int row = 0; row < rowCount; row++) {
            accumulateWeightedSlice(out, outRow, outOffset, rows, rowOffset + row, rowColumnOffset, width,
                    weights[weightsOffset + row]);
        }
    }

    default boolean usesOptimizedAccumulateWeightedRows(AbstractTensor out, AbstractTensor rows) {
        return false;
    }

    /** Multiplies one row slice by {@code factor}. */
    default void normalizeSlice(AbstractTensor tensor, int row, int offset, int length, float factor) {
        for (int col = 0; col < length; col++) {
            tensor.set(tensor.get(row, offset + col) * factor, row, offset + col);
        }
    }

    default boolean usesOptimizedNormalizeSlice(AbstractTensor tensor) {
        return false;
    }

    default void scaleSlice(AbstractTensor tensor, int row, int offset, int length, float factor) {
        normalizeSlice(tensor, row, offset, length, factor);
    }

    default boolean usesOptimizedScaleSlice(AbstractTensor tensor) {
        return usesOptimizedNormalizeSlice(tensor);
    }

    default AbstractTensor activationMultiplyQuantize(AbstractTensor gate, AbstractTensor up,
            ActivationFunction.Type activation, DType qtype, int offset, int length) {
        Preconditions.checkArgument(gate.shape().equals(up.shape()), "gate and up must have same shape");
        try (FloatBufferTensor hidden = new FloatBufferTensor(gate.shape())) {
            for (int row = 0; row < gate.shape().first(); row++) {
                for (int col = offset; col < offset + length; col++) {
                    hidden.set(ActivationFunction.eval(activation, gate.get(row, col)) * up.get(row, col), row, col);
                }
            }
            return quantize(hidden, qtype, offset, length);
        }
    }

    /**
     * Computes one-token decode attention over paged KV cache tensors without packing the full visible KV window.
     *
     * <p>The default implementation uses existing provider kernels over the current pages: one page-aware QK pass,
     * softmax, then one page-aware value accumulation pass. It avoids model-level orchestration and avoids packing the
     * full visible KV window.</p>
     */
    default void decodePagedAttention(AbstractTensor valueOut, AbstractTensor query, AbstractTensor[] keyPages,
            AbstractTensor[] valuePages, int visibleRows, int numberOfHeads, int numberOfKeyValueHeads, int headSize,
            float scale, Float softcap) {
        for (int head = 0; head < numberOfHeads; head++) {
            decodePagedAttentionHeadWithProviderKernels(valueOut, query, keyPages, valuePages, visibleRows,
                    numberOfHeads, numberOfKeyValueHeads, headSize, scale, softcap, head);
        }
    }

    default boolean supportsDecodePagedAttention(AbstractTensor valueOut, AbstractTensor query, AbstractTensor[] keyPages,
            AbstractTensor[] valuePages, int visibleRows, int numberOfHeads, int numberOfKeyValueHeads, int headSize,
            float scale, Float softcap) {
        return true;
    }

    default boolean supportsFlashDecodePagedAttention(AbstractTensor valueOut, AbstractTensor query,
            AbstractTensor[] keyPages, AbstractTensor[] valuePages, int visibleRows, int numberOfHeads,
            int numberOfKeyValueHeads, int headSize, float scale, Float softcap) {
        return false;
    }

    default void flashDecodePagedAttention(AbstractTensor valueOut, AbstractTensor query, AbstractTensor[] keyPages,
            AbstractTensor[] valuePages, int visibleRows, int numberOfHeads, int numberOfKeyValueHeads, int headSize,
            float scale, Float softcap) {
        throw new UnsupportedOperationException("flashDecodePagedAttention is not supported by " + name());
    }

    default void decodePagedAttentionHeadWithProviderKernels(AbstractTensor valueOut, AbstractTensor query,
            AbstractTensor[] keyPages, AbstractTensor[] valuePages, int visibleRows, int numberOfHeads,
            int numberOfKeyValueHeads, int headSize, float scale, Float softcap, int head) {
        Preconditions.checkArgument(keyPages.length == valuePages.length, "key/value page count mismatch");
        Preconditions.checkArgument(query.shape().first() == 1, "decode query must have one row");
        Preconditions.checkArgument(valueOut.shape().first() == 1, "decode value output must have one row");
        int headGroupSize = numberOfHeads / numberOfKeyValueHeads;
        int kvHead = head / headGroupSize;
        int queryOffset = head * headSize;
        int kvOffset = kvHead * headSize;
        try (AbstractTensor attn = new FloatBufferTensor(1, visibleRows)) {
            int globalRow = 0;
            for (AbstractTensor keyPage : keyPages) {
                if (globalRow >= visibleRows) {
                    break;
                }
                int rows = (int) Math.min(keyPage.shape().first(), visibleRows - globalRow);
                batchDotProduct(attn, query, keyPage, queryOffset, kvOffset, headSize, globalRow, 0, rows);
                globalRow += rows;
            }
            VectorTensorMathUtils.scaledSoftMax(attn, 0, visibleRows, scale, softcap);
            globalRow = 0;
            for (AbstractTensor valuePage : valuePages) {
                if (globalRow >= visibleRows) {
                    break;
                }
                int rows = (int) Math.min(valuePage.shape().first(), visibleRows - globalRow);
                saxpy(attn, valuePage, valueOut, kvOffset, queryOffset, headSize, globalRow, 0, rows);
                globalRow += rows;
            }
        }
    }

    /**
     * Collects the total sum of each position in the tensor.  (For testing purposes)
     */
    default float sum(AbstractTensor a) {
        float sum = 0f;
        int[] cursor = new int[a.dims()];
        while (a.iterate(cursor))
            sum += a.get(cursor);
        return sum;
    }
}
