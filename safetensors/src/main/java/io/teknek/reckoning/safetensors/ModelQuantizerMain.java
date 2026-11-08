package io.teknek.reckoning.safetensors;

import io.teknek.reckoning.DType;
import io.teknek.reckoning.tensor2.Lighter;

public final class ModelQuantizerMain {
    private ModelQuantizerMain() {
    }

    public static void main(String[] args) {
        if (args.length < 4 || args.length > 5) {
            throw new IllegalArgumentException("usage: ModelQuantizerMain <inputOwner> <inputModel> <outputOwner> <outputModel> [Q4|I8|BF16|F32]");
        }
        DType targetType = args.length == 5 ? DType.valueOf(args[4].toUpperCase()) : DType.Q4;
        Lighter lighter = new Lighter();
        new ModelQuantizer(lighter).quantizeCachedModel(args[0], args[1], args[2], args[3], targetType,
                ModelQuantizer.DEFAULT_Q4_TENSOR_FILTER);
    }
}
