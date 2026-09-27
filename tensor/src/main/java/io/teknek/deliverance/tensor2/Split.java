package io.teknek.deliverance.tensor2;

import com.google.common.base.Preconditions;
import io.teknek.deliverance.tensor.TensorShape;

public class Split {
    private TensorRef source;
    private TensorRef[] destinations;
    private int chunks;
    private int dimension;

    public Split source(TensorRef source) {
        this.source = source;
        return this;
    }

    public Split into(TensorRef[] destinations) {
        this.destinations = destinations;
        return this;
    }

    public Split chunks(int chunks) {
        this.chunks = chunks;
        return this;
    }

    public Split dimension(int dimension) {
        this.dimension = dimension;
        return this;
    }

    public TensorRef getSource() { return source; }
    public TensorRef[] getDestinations() { return destinations; }
    public int getChunks() { return chunks; }
    public int getDimension() { return dimension; }

    public static TensorRef[] allocate(Lighter lighter, TensorRef source, int chunks, int dimension) {
        Preconditions.checkArgument(source != null, "Source tensor must be set");
        Preconditions.checkArgument(chunks > 0, "Chunk count must be positive");
        Preconditions.checkArgument(dimension >= 0 && dimension < source.dims(), "Split dimension out of bounds");
        int dimensionSize = source.shape().dim(dimension);
        Preconditions.checkArgument(dimensionSize % chunks == 0, "Chunks must be of equal size");
        int[] chunkShape = source.shape().shapeArray();
        chunkShape[dimension] = dimensionSize / chunks;
        TensorRef[] result = new TensorRef[chunks];
        try {
            for (int chunk = 0; chunk < chunks; chunk++) {
                result[chunk] = lighter.allocate(source.dType(), TensorShape.of(chunkShape));
            }
            return result;
        } catch (RuntimeException | Error e) {
            for (TensorRef tensor : result) {
                if (tensor != null) {
                    tensor.close();
                }
            }
            throw e;
        }
    }
}
