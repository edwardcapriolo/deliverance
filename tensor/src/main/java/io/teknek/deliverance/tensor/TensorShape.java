package io.teknek.deliverance.tensor;

import com.google.common.base.Preconditions;

import java.util.Arrays;
import java.util.Objects;
import java.util.Optional;

/**
 * Logical shape plus optional sparse residency metadata for a tensor.
 *
 * <p>The {@code tshape} array always describes the full logical tensor dimensions. A tensor may still store only
 * a contiguous row window, a contiguous column window, or neither. In this file the word <em>resident</em>
 * means "physically present in this tensor's current backing storage".</p>
 *
 * <p>Example: a tensor can have logical shape {@code [4096, 14336]} while only rows {@code [1024, 1536)} are
 * resident in memory. Callers still reason about logical coordinates, and {@code TensorShape} translates those
 * logical coordinates into flat offsets within the resident storage.</p>
 */
public class TensorShape {
    /** Tensors are always at least 2D a single tensor of a single element 1 thus becomes [[1]] */
    public static TensorShape ONE = of(1, 1);

    public static TensorShape of(int... shape) {
        if (shape.length == 1) {
            shape = new int[] { 1, shape[0] };
        }
        return new TensorShape(shape, Optional.empty(), Optional.empty());
    }

    /**
     * Creates a shape whose last dimension is backed by only a sparse window of the logical columns.
     *
     * <p>The {@code shape} array still describes the full logical tensor shape. The sparse offset then says
     * which contiguous column window is actually resident in memory.</p>
     *
     * <p>Example: a logical shape of {@code [4, 128]} with {@code SparseOffset.of(32, 64)} means:</p>
     *
     * <ul>
     * <li>logical tensor shape is still 4 rows by 128 columns</li>
     * <li>only columns {@code [32, 96)} are physically present</li>
     * <li>the resident column window therefore has offset {@code 32} and length {@code 64}</li>
     * </ul>
     *
     * <p>Despite the {@code SparseOffset} accessor name {@code getEnd()}, this code interprets the second
     * value as a length.</p>
     */
    public static TensorShape sparseColumn(int[] shape, SparseOffset<Integer> sparseOffset) {
        return new TensorShape(shape, Optional.empty(), Optional.of(sparseOffset));
    }

    /**
     * Creates a shape whose second-to-last dimension is backed by only a sparse window of the logical rows.
     *
     * <p>The {@code shape} array still describes the full logical tensor shape. The sparse offset then says
     * which contiguous row window is actually resident in memory.</p>
     *
     * <p>Example: a logical shape of {@code [4096, 14336]} with {@code SparseOffset.of(1024, 512)} means:</p>
     *
     * <ul>
     * <li>logical tensor shape is still 4096 rows by 14336 columns</li>
     * <li>only rows {@code [1024, 1536)} are physically present</li>
     * <li>the resident row window therefore has offset {@code 1024} and length {@code 512}</li>
     * </ul>
     *
     * <p>This is commonly used for sharded or row-sliced weight tensors. Despite the {@code SparseOffset}
     * accessor name {@code getEnd()}, this code interprets the second value as a length.</p>
     */
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

    /**
     * Builds a tensor shape that may represent either a fully dense tensor or a dense logical tensor backed by
     * only a contiguous sparse row/column window.
     *
     * <p>{@code tshape} always stores the full logical shape. The sparse row/column fields describe which part
     * of that logical tensor is actually resident in memory. The derived {@code sparse*Offset} values are the
     * logical starting coordinates of the resident window. The derived {@code sparse*Length} values are the
     * resident sizes, not exclusive end positions.</p>
     */
    private TensorShape(int[] shape, Optional<SparseOffset<Integer>> sparseRowWindow, Optional<SparseOffset<Integer>> sparseColumnWindow) {
        Preconditions.checkArgument(
                shape.length > 1,
                "Shape must have at least two dimensions, even if first is 1 (to represent a vector)"
        );

        this.tshape = shape;
        this.sparseColumnRange = sparseColumnWindow;
        this.sparseRowRange = sparseRowWindow;
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

    /**
     * Converts a logical row/column coordinate into a flat offset within the resident backing storage.
     *
     * <p>If this shape is sparse, {@code row} and {@code column} are still expressed in logical tensor
     * coordinates, not local coordinates relative to the resident window. The sparse offsets are subtracted here
     * so callers can continue to reason in logical tensor space.</p>
     */
    public final int getOffset(int row, int column){
        return sparseColumnLength * (row - sparseRowOffset) + column - sparseColumnOffset;
    }

    /**
     * Note: This method will return positions outside the tensor
     * @param pdims one or more dimenstions
     * @return the position in the 1 Demensional view of the tensor
     */
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


    public int [] shapeArray(){
        return Arrays.copyOf(tshape, tshape.length);
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

    /**
     * Returns a new shape whose last dimension has been scaled by the given factor.
     *
     * <p>This is currently used by Q4 tensor layouts, where one logical value shape is represented by a
     * different packed-storage shape. For example, Q4 stores 32 logical values in 16 bytes, so the stored last
     * dimension is half the logical width and this helper is used to move between those two views.</p>
     *
     * <p>If the shape has a sparse resident column window, that window is scaled too. In other words, both the
     * last dimension and the sparse column {@code (offset, length)} are multiplied by {@code scale}.</p>
     *
     * @param scale factor applied to the last dimension, and to any sparse resident column window.
     * @return a new shape with the scaled last dimension.
     */
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

    /**
     *
     * @return the size of the first dimension of the tensor [2.4] -> 2
     */
    public int first() {
        return tshape[0];
    }

    /**
     *
     * @return the size of the last dimension of the tensor [2.4] -> 4
     */
    public int last() {
        return tshape[tshape.length - 1];
    }

    public long size() {
        return capacity;
    }

    /**
     * Returns a new shape that keeps the same logical dimensions but marks only a contiguous last-dimension
     * column window as resident.
     *
     * <p>The arguments are interpreted as {@code (columnOffset, columnLength)}. Here, resident means those are
     * the columns physically stored in the current tensor buffer.</p>
     */
    public TensorShape sparsifyColumns(int offset, int length) {
        Preconditions.checkArgument(!isSparse, "Cannot sparsify a sparse tensor");
        return new TensorShape(tshape, Optional.empty(), Optional.of(SparseOffset.of(offset, length)));
    }

    public TensorShape slice(int numDims) {
        Preconditions.checkArgument(numDims < tshape.length, "Too many dimensions specified for tensor");
        int newLength = tshape.length - numDims;
        if (newLength == 1) {
            return new TensorShape(new int[] { 1, tshape[tshape.length - 1] }, sparseRowRange, sparseColumnRange);
        }
        return new TensorShape(Arrays.copyOfRange(tshape, numDims, tshape.length), sparseRowRange, sparseColumnRange);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        TensorShape that = (TensorShape) o;
        // TODO: sparseRowRange is intentionally ignored here to preserve existing behavior during doc/comment work.
        // Revisit in a dedicated behavior-change pass after checking map/cache/equality call sites.
        return Arrays.equals(tshape, that.tshape) && Objects.equals(sparseColumnRange, that.sparseColumnRange);
    }

    @Override
    public int hashCode() {
        // TODO: keep hashCode aligned with the current equals() behavior until sparseRowRange can be safely added.
        int result = Objects.hash(sparseColumnRange);
        result = 31 * result + Arrays.hashCode(tshape);
        return result;
    }

    @Override
    public String toString() {
        return "TensorShape{" + "tshape=" + Arrays.toString(tshape) + ", capacity=" + capacity + ", sparseRange=" + sparseColumnRange + '}';
    }
}
