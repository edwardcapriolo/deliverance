package io.teknek.deliverance.safetensors;

import io.teknek.deliverance.tensor.TensorInfo;

import java.io.File;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Map;

/** Command-line utility for inspecting safetensors model directories with Deliverance's Java loader. */
public final class SafetensorsInspector {
    private SafetensorsInspector() {
    }

    public static void main(String[] args) {
        if (args.length != 1) {
            throw new IllegalArgumentException("Usage: SafetensorsInspector <model-directory>");
        }
        inspect(new File(args[0]));
    }

    public static void inspect(File modelDirectory) {
        try (DefaultWeightLoader loader = new DefaultWeightLoader(modelDirectory)) {
            System.out.println("modelRoot=" + loader.modelRoot().orElse(modelDirectory.toPath()));
            System.out.println("modelDType=" + loader.getModelDType());
            System.out.println("tensorCount=" + loader.tensorInfoMap().size());
            loader.tensorInfoMap().entrySet().stream()
                    .sorted(Comparator.comparing(Map.Entry::getKey))
                    .forEach(entry -> print(entry.getKey(), entry.getValue()));
        }
    }

    private static void print(String name, TensorInfo info) {
        System.out.printf("%s dtype=%s shape=%s offsets=%s%n", name, info.dType,
                Arrays.toString(info.shape), Arrays.toString(info.dataOffsets));
    }
}
