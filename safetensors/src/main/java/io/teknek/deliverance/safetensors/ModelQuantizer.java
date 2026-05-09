package io.teknek.deliverance.safetensors;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.AbstractTensorUtils;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.Predicate;
import java.util.stream.Stream;

public class ModelQuantizer {
    private final long maxShardSize;

    public static final Predicate<String> DEFAULT_Q4_TENSOR_FILTER = name ->
            !name.endsWith(".qb")
                    && name.endsWith(".weight")
                    // Keep embedding tables and lm_head dense across both plain and wrapped model roots
                    // (for example model.embed_tokens.weight vs model.language_model.embed_tokens.weight).
                    && !name.endsWith("embed_tokens.weight")
                    && !name.endsWith("embed_tokens_per_layer.weight")
                    && !name.endsWith("lm_head.weight");

    public ModelQuantizer() {
        this(SafeTensorWriter.DEFAULT_MAX_SHARD_SIZE);
    }

    public ModelQuantizer(long maxShardSize) {
        this.maxShardSize = maxShardSize;
    }

    public void quantizeCachedModel(String inputOwner, String inputModel, String outputOwner, String outputModel) {
        quantizeCachedModel(inputOwner, inputModel, outputOwner, outputModel, DType.Q4, DEFAULT_Q4_TENSOR_FILTER);
    }

    public void quantizeCachedModel(String inputOwner, String inputModel, String outputOwner, String outputModel,
            DType targetType, Predicate<String> tensorFilter) {
        ModelFetcher sourceFetcher = new ModelFetcher(inputOwner, inputModel);
        ModelFetcher outputFetcher = new ModelFetcher(outputOwner, outputModel);
        Path sourceDir = sourceFetcher.pathForModel();
        Path outputDir = outputFetcher.pathForModel();
        if (!Files.exists(sourceDir)) {
            throw new IllegalArgumentException("Input model not found in cache: " + sourceDir);
        }
        quantizeModelDirectory(sourceDir, outputDir, targetType, tensorFilter);
    }

    public void quantizeModelDirectory(Path sourceDir, Path outputDir) {
        quantizeModelDirectory(sourceDir, outputDir, DType.Q4, DEFAULT_Q4_TENSOR_FILTER);
    }

    public void quantizeModelDirectory(Path sourceDir, Path outputDir, DType targetType, Predicate<String> tensorFilter) {
        if (targetType != DType.Q4 && targetType != DType.I8 && targetType != DType.BF16 && targetType != DType.F32) {
            throw new IllegalArgumentException("Unsupported export target " + targetType);
        }
        try {
            Files.createDirectories(outputDir);
            copyNonWeightFiles(sourceDir, outputDir);
            try (DefaultWeightLoader loader = new DefaultWeightLoader(sourceDir.toFile())) {
                Map<String, AbstractTensor> converted = new LinkedHashMap<>();
                try {
                    for (String name : loader.tensorInfoMap().keySet()) {
                        if (name.endsWith(".qb") || isLogicalSplitTensor(name, loader)) {
                            continue;
                        }
                        AbstractTensor original = loader.load(name);
                        AbstractTensor tensor = original;
                        if (tensorFilter.test(name) && canQuantize(original, targetType)) {
                            AbstractTensor quantized = AbstractTensorUtils.quantize(original, targetType, true);
                            if (quantized != original) {
                                tensor = quantized;
                                original.close();
                            }
                        }
                        converted.put(name, tensor);
                    }
                    SafeTensorWriter.writeModel(outputDir, loader.metadata(), converted, maxShardSize);
                } finally {
                    converted.values().forEach(AbstractTensor::close);
                }
            }
        } catch (IOException e) {
            throw new RuntimeException("Unable to quantize model from " + sourceDir + " to " + outputDir, e);
        }
    }

    /**
     * Q4/I8 export currently only supports matrix-like tensors. Non-2D tensors such as convolution
     * kernels and scalar calibration tensors are kept dense to avoid invalid block-shape handling.
     */
    boolean canQuantize(AbstractTensor tensor, DType targetType) {
        return tensor.dType() != targetType && tensor.shape().dims() == 2;
    }

    /**
     * DefaultWeightLoader exposes both logical tensor names and internal `-part-*` chunks for
     * oversized tensors. The logical parent is not directly loadable, so the quantizer must skip
     * it and process the concrete parts instead.
     */
    boolean isLogicalSplitTensor(String name, DefaultWeightLoader loader) {
        return !name.contains("-part-") && loader.tensorInfoMap().containsKey(name + "-part-0");
    }

    private void copyNonWeightFiles(Path sourceDir, Path outputDir) throws IOException {
        try (Stream<Path> paths = Files.walk(sourceDir)) {
            paths.filter(Files::isRegularFile)
                    .filter(path -> !isWeightFile(path))
                    .forEach(path -> copyRelative(path, sourceDir, outputDir));
        }
    }

    private boolean isWeightFile(Path path) {
        String fileName = path.getFileName().toString();
        return fileName.endsWith(".safetensors") || fileName.equals(SafeTensorIndexPojo.MODEL_INDEX_JSON);
    }

    private void copyRelative(Path sourceFile, Path sourceDir, Path outputDir) {
        try {
            Path relative = sourceDir.relativize(sourceFile);
            Path target = outputDir.resolve(relative);
            Files.createDirectories(target.getParent());
            Files.copy(sourceFile, target, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.COPY_ATTRIBUTES);
        } catch (IOException e) {
            throw new RuntimeException("Unable to copy metadata file " + sourceFile, e);
        }
    }
}
