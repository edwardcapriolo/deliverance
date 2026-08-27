package io.teknek.deliverance.tensor.operations;

import com.google.common.base.Preconditions;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.TensorMutability;
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
     * Writes {@code output = exp(input)} over a contiguous window for every row in the input tensor.
     *
     * <p>This is a first-class tensor operation because probability code such as softmax and diffusion entropy needs
     * exponentials over large logits tensors. Implementations should use their best available backend; Java Vector API
     * providers may still use scalar exponentials until a native vector exp is available.</p>
     */
    default void exp(AbstractTensor input, AbstractTensor output, int offset, int length) {
        TensorMutability.requireWritable(output, "exp");
        Preconditions.checkArgument(input.shape().equals(output.shape()), "input and output must have same shape");
        int limit = offset + length;
        for (int row = 0; row < input.shape().first(); row++) {
            for (int i = offset; i < limit; i++) {
                output.set((float) net.jafama.FastMath.exp(input.get(row, i)), row, i);
            }
        }
    }

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

    default AbstractTensor activationMultiplyQuantize(AbstractTensor gate, AbstractTensor up,
            ActivationFunction.Type activation, DType qtype, int offset, int length) {
        Preconditions.checkArgument(gate.shape().equals(up.shape()), "gate and up must have same shape");
        FloatBufferTensor hidden = new FloatBufferTensor(gate.shape());
        try {
            for (int row = 0; row < gate.shape().first(); row++) {
                for (int col = offset; col < offset + length; col++) {
                    hidden.set(ActivationFunction.eval(activation, gate.get(row, col)) * up.get(row, col), row, col);
                }
            }
            if (hidden.dType() == qtype) {
                return hidden;
            }
            return quantize(hidden, qtype, offset, length);
        } catch (RuntimeException | Error e) {
            hidden.close();
            throw e;
        }
    }

    /**
     * Gathers BERT embedding rows and adds word, token-type, and position embeddings into {@code output}.
     *
     * <pre>{@code
     * output[row, col] = wordEmbeddings[inputIds[row], col]
     *                  + tokenTypeEmbeddings[tokenTypeIds[row], col]
     *                  + positionEmbeddings[positionIds[row], col]
     * }</pre>
     */
    default void gatherRowsAdd(AbstractTensor output, AbstractTensor wordEmbeddings, int[] inputIds,
            AbstractTensor tokenTypeEmbeddings, int[] tokenTypeIds, AbstractTensor positionEmbeddings,
            int[] positionIds, int rowOffset, int rowCount) {
        TensorMutability.requireWritable(output, "gatherRowsAdd");
        Preconditions.checkArgument(output.dims() == 2, "output must be 2D");
        Preconditions.checkArgument(wordEmbeddings.dims() == 2, "wordEmbeddings must be 2D");
        Preconditions.checkArgument(tokenTypeEmbeddings.dims() == 2, "tokenTypeEmbeddings must be 2D");
        Preconditions.checkArgument(positionEmbeddings.dims() == 2, "positionEmbeddings must be 2D");
        int rows = (int) output.shape().first();
        int hidden = (int) output.shape().last();
        Preconditions.checkArgument(inputIds.length == rows, "inputIds length must match output rows");
        Preconditions.checkArgument(tokenTypeIds.length == rows, "tokenTypeIds length must match output rows");
        Preconditions.checkArgument(positionIds.length == rows, "positionIds length must match output rows");
        Preconditions.checkArgument(wordEmbeddings.shape().last() == hidden, "word hidden size mismatch");
        Preconditions.checkArgument(tokenTypeEmbeddings.shape().last() == hidden, "token type hidden size mismatch");
        Preconditions.checkArgument(positionEmbeddings.shape().last() == hidden, "position hidden size mismatch");
        Preconditions.checkArgument(rowOffset >= 0 && rowCount >= 0 && rowOffset + rowCount <= rows,
                "row range out of bounds");
        int rowLimit = rowOffset + rowCount;
        for (int row = rowOffset; row < rowLimit; row++) {
            int inputId = inputIds[row];
            int tokenTypeId = tokenTypeIds[row];
            int positionId = positionIds[row];
            Preconditions.checkArgument(inputId >= 0 && inputId < wordEmbeddings.shape().first(),
                    "input id out of bounds");
            Preconditions.checkArgument(tokenTypeId >= 0 && tokenTypeId < tokenTypeEmbeddings.shape().first(),
                    "token type id out of bounds");
            Preconditions.checkArgument(positionId >= 0 && positionId < positionEmbeddings.shape().first(),
                    "position id out of bounds");
            for (int col = 0; col < hidden; col++) {
                output.set(wordEmbeddings.get(inputId, col)
                        + tokenTypeEmbeddings.get(tokenTypeId, col)
                        + positionEmbeddings.get(positionId, col), row, col);
            }
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
