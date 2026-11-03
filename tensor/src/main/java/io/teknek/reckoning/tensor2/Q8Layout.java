package io.teknek.reckoning.tensor2;

import com.google.common.base.Preconditions;
import io.teknek.reckoning.tensor.TensorShape;

final class Q8Layout {
    static final int BLOCK_SIZE = 32;
    static final String SCALE_SIDECAR = "q8.scale";

    private Q8Layout() {
    }

    static TensorRef scale(TensorRef tensor) {
        return tensor.sidecar(SCALE_SIDECAR);
    }

    static int scaleColumn(int column) {
        return column / BLOCK_SIZE;
    }

    static TensorShape scaleShape(TensorShape shape) {
        Preconditions.checkArgument(shape.last() % BLOCK_SIZE == 0,
                "I8 tensor last dimension must be a multiple of %s", BLOCK_SIZE);
        Preconditions.checkArgument(shape.sparseColumnOffset() % BLOCK_SIZE == 0,
                "I8 sparse column offset must be a multiple of %s", BLOCK_SIZE);
        Preconditions.checkArgument(shape.sparseColumnLength() % BLOCK_SIZE == 0,
                "I8 sparse column length must be a multiple of %s", BLOCK_SIZE);
        int[] scaleShape = shape.shapeArray();
        scaleShape[scaleShape.length - 1] = scaleColumn(scaleShape[scaleShape.length - 1]);
        if (shape.sparseColumnOffset() != 0 || shape.sparseColumnLength() != shape.last()) {
            return TensorShape.sparseColumn(scaleShape, io.teknek.reckoning.tensor.SparseOffset.of(
                    scaleColumn(shape.sparseColumnOffset()),
                    scaleColumn(shape.sparseColumnLength())));
        }
        return TensorShape.of(scaleShape);
    }
}
