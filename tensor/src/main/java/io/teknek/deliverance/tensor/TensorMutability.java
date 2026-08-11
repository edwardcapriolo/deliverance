package io.teknek.deliverance.tensor;

public final class TensorMutability {
    private TensorMutability() {
    }

    public static void requireWritable(AbstractTensor tensor, String operation) {
        if (tensor instanceof ReadOnlyTensor) {
            throw new UnsupportedOperationException(operation + " requires writable tensor");
        }
    }

    public static AbstractTensor unwrapReadOnly(AbstractTensor tensor) {
        while (tensor instanceof ReadOnlyTensor readOnly) {
            tensor = readOnly.delegate();
        }
        return tensor;
    }
}
