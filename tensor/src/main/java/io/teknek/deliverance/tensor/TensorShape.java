package io.teknek.deliverance.tensor;

import com.google.common.base.Preconditions;

import java.util.Arrays;
import java.util.Objects;
import java.util.Optional;

public class TensorShape {
    /** Tensors are always two dimentation a single tensor of a single element 1 thus becomes [[1]] */
    public static TensorShape ONE = of(1, 1);

    public static TensorShape of(int... shape) {
        if (shape.length == 1) {
            shape = new int[] { 1, shape[0] };
        }
        return new TensorShape(shape, Optional.empty(), Optional.empty());
    }

    public static TensorShape sparseColumn(int[] shape, SparseOffset<Integer> sparseOffset) {
        return new TensorShape(shape, Optional.empty(), Optional.of(sparseOffset));
    }

    public static TensorShape sparseRow(int[] shape, SparseOffset<Integer> sparseOffset) {
        return new TensorShape(shape, Optional.of(sparseOffset), Optional.empty());
    }

    private final int[] tshape;
    private final long capacity;

    private final Optional<SparseOffset<Integer>> sparseColumnRange;
    private final Optional<SparseOffset<Integer>> sparseRowRange;
    private final boolean isSparse;
    private final int sparseColumnOffset;
    private final int sparseColumnLength;
    private final int sparseRowOffset;
    private final int sparseRowLength;

    private TensorShape(int[] shape, Optional<SparseOffset<Integer>> sparseRowRange, Optional<SparseOffset<Integer>> sparseColumnRange) {
        Preconditions.checkArgument(
                shape.length > 1,
                "Shape must have at least two dimensions, even if first is 1 (to represent a vector)"
        );

        this.tshape = shape;
        this.sparseColumnRange = sparseColumnRange;
        this.sparseRowRange = sparseRowRange;
        this.isSparse = this.sparseColumnRange.isPresent() || this.sparseRowRange.isPresent();

        this.sparseColumnOffset = this.sparseColumnRange.map(SparseOffset::getStart).orElse(0);
        this.sparseColumnLength = this.sparseColumnRange.map(SparseOffset::getEnd).orElse(shape[shape.length - 1]);

        this.sparseRowOffset = this.sparseRowRange.map(SparseOffset::getStart).orElse(0);
        this.sparseRowLength = this.sparseRowRange.map(SparseOffset::getEnd).orElse(shape[shape.length - 2]);

        long c = 1;
        for (int i = 0; i < shape.length - 2; i++)
            c *= shape[i];

        c *= sparseRowLength;
        c *= sparseColumnLength;
        this.capacity = c;
    }

    public final boolean isSparse() {
        return isSparse;
    }

    public int dims() {
        return tshape.length;
    }

    public int dim(int i) {
        Preconditions.checkArgument(i < tshape.length);
        return tshape[i];
    }

    /**
     *
     * @return Option.some if  the input dimensions can exist the shape Optional.empty otherwise
     */
    public Optional<Integer> safeOffset(int dimension1, int dimension2){
        //This could have been generalized. It is quite sad I wrote it this way
        if (dims() != 2){
            return Optional.empty();
        }
        if(dimension1 < 0 || dimension2 < 0){
            return Optional.empty();
        }
        if (dimension1 >= this.dim(0)){
            return Optional.empty();
        }
        if (dimension2 >= this.dim(1)){
            return Optional.empty();
        }
        return Optional.of(sparseColumnLength * (dimension1 - sparseRowOffset) + dimension2 - sparseColumnOffset);
    }

    public final int getOffset(int... pdims) {
        switch (pdims.length) {
            case 1:
                return sparseColumnLength * (pdims[0] - sparseRowOffset) - sparseColumnOffset;
            case 2:
                return sparseColumnLength * (pdims[0] - sparseRowOffset) + pdims[1] - sparseColumnOffset; // Most common case
            case 3:
                return (sparseColumnLength * tshape[1] * (pdims[0] - sparseRowOffset)) + (sparseColumnLength * pdims[1]) + pdims[2]
                        - sparseColumnOffset;
            default:
                int totalOffset = 0;
                for (int d = 0; d < pdims.length - 1; ++d) { // Stop before last dimension
                    int offset = sparseColumnLength;
                    for (int i = tshape.length - 2; i > d; --i) { // factor scaling of each dim shape
                        offset *= tshape[i];
                    }

                    totalOffset += pdims[d] * offset;
                }

                return totalOffset + pdims[pdims.length - 1] - sparseColumnOffset;
        }
    }

    public int sparseColumnLength() {
        return sparseColumnLength;
    }

    public int sparseColumnOffset() {
        return sparseColumnOffset;
    }

    public int sparseRowLength() {
        return sparseRowLength;
    }

    public int sparseRowOffset() {
        return sparseRowOffset;
    }

    //used only in q4 tenso
    public TensorShape scaleLastDim(float scale) {
        int[] copy = Arrays.copyOf(tshape, tshape.length);
        copy[copy.length - 1] *= scale;
        return sparseColumnRange.isPresent()
                ? sparseColumn(copy, SparseOffset.of((int) (sparseColumnOffset * scale), (int) (sparseColumnLength * scale)))
                : of(copy);
    }

    //used by split tensor in abstract tensor
    public TensorShape setDimValue(int dim, int value) {
        Preconditions.checkArgument(dim < tshape.length);
        int[] copy = Arrays.copyOf(tshape, tshape.length);
        copy[dim] = value;
        int newSparseLength = copy[copy.length - 1];
        return sparseColumnRange.isPresent() ? sparseColumn(copy, SparseOffset.of(sparseColumnOffset, newSparseLength)) : of(copy);
    }

    public int first() {
        return tshape[0];
    }

    public int last() {
        return tshape[tshape.length - 1];
    }

    public long size() {
        return capacity;
    }

    public TensorShape sparsifyColumns(int offset, int length) {
        Preconditions.checkArgument(!isSparse, "Cannot sparsify a sparse tensor");
        return new TensorShape(tshape, Optional.empty(), Optional.of(SparseOffset.of(offset, length)));
    }

    public TensorShape slice(int numDims) {
        Preconditions.checkArgument(numDims < tshape.length, "Too many dimensions specified for tensor");
        int newLength = tshape.length - numDims;
        if (newLength == 1) return new TensorShape(new int[] { 1, tshape[tshape.length - 1] }, sparseRowRange, sparseColumnRange);

        return new TensorShape(Arrays.copyOfRange(tshape, numDims, tshape.length), sparseRowRange, sparseColumnRange);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        TensorShape that = (TensorShape) o;
        return Arrays.equals(tshape, that.tshape) && Objects.equals(sparseColumnRange, that.sparseColumnRange);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(sparseColumnRange);
        result = 31 * result + Arrays.hashCode(tshape);
        return result;
    }

    @Override
    public String toString() {
        return "TensorShape{" + "tshape=" + Arrays.toString(tshape) + ", capacity=" + capacity + ", sparseRange=" + sparseColumnRange + '}';
    }
}