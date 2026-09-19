package io.teknek.deliverance.tensor2;

import com.google.common.base.Preconditions;
import io.teknek.deliverance.tensor.SparseOffset;
import io.teknek.deliverance.tensor.TensorShape;

final class Q4Layout {
    static final int BLOCK_SIZE = 32;
    static final int HALF_BLOCK = BLOCK_SIZE / 2;
    static final String SCALE_SIDECAR = "q4.scale";

    private Q4Layout() {
    }

    static TensorRef scale(TensorRef tensor) {
        return tensor.sidecar(SCALE_SIDECAR);
    }

    static int blockIndex(int column) {
        return column / BLOCK_SIZE;
    }

    static TensorShape scaleShape(TensorShape shape) {
        Preconditions.checkArgument(shape.last() % BLOCK_SIZE == 0,
                "Q4 tensor last dimension must be a multiple of %s", BLOCK_SIZE);
        Preconditions.checkArgument(shape.sparseColumnOffset() % BLOCK_SIZE == 0,
                "Q4 sparse column offset must be a multiple of %s", BLOCK_SIZE);
        Preconditions.checkArgument(shape.sparseColumnLength() % BLOCK_SIZE == 0,
                "Q4 sparse column length must be a multiple of %s", BLOCK_SIZE);
        int[] scaleShape = shape.shapeArray();
        scaleShape[scaleShape.length - 1] = blockIndex(scaleShape[scaleShape.length - 1]);
        if (shape.sparseColumnOffset() != 0 || shape.sparseColumnLength() != shape.last()) {
            return TensorShape.sparseColumn(scaleShape, SparseOffset.of(
                    blockIndex(shape.sparseColumnOffset()),
                    blockIndex(shape.sparseColumnLength())));
        }
        return TensorShape.of(scaleShape);
    }
}
