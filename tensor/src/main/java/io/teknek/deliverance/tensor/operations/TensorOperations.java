package io.teknek.deliverance.tensor.operations;

import com.google.common.base.Preconditions;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorShape;
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
     * Quantizes the tensor to the specified type (if supported)
     */
    /*
    default AbstractTensor quantize(AbstractTensor t, DType qtype, int offset, int length) {
        AbstractTensor t2 = TensorCache.instance.get(t.dType(), t.shape());
        t2.copyFrom(t, offset, offset, length);
        return t2;
    }*/
    AbstractTensor quantize(AbstractTensor t, DType qtype, int offset, int length);

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
