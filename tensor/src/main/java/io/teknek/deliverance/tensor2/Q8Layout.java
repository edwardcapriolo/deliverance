package io.teknek.deliverance.tensor2;

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
}
