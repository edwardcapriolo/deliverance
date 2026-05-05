package io.teknek.deliverance.safetensors;

import io.teknek.deliverance.DType;

public final class ModelQuantizerMain {
    private ModelQuantizerMain() {
    }

    public static void main(String[] args) {
        if (args.length < 4 || args.length > 5) {
            throw new IllegalArgumentException("usage: ModelQuantizerMain <inputOwner> <inputModel> <outputOwner> <outputModel> [Q4|I8|BF16|F32]");
        }
        DType targetType = args.length == 5 ? DType.valueOf(args[4].toUpperCase()) : DType.Q4;
        new ModelQuantizer().quantizeCachedModel(args[0], args[1], args[2], args[3], targetType,
                ModelQuantizer.DEFAULT_Q4_TENSOR_FILTER);
    }
}
