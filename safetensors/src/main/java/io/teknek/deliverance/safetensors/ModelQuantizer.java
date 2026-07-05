package io.teknek.deliverance.safetensors;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.AbstractTensorUtils;
import io.teknek.deliverance.tensor.TensorInfo;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.StandardOpenOption;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Comparator;
import java.util.function.Predicate;
import java.util.stream.Stream;

import static io.teknek.deliverance.tensor.impl.Q4ByteBufferTensor.BLOCK_SIZE;
import static io.teknek.deliverance.tensor.impl.Q4ByteBufferTensor.HALF_BLOCK;

public class ModelQuantizer {
    private static final Logger LOGGER = LoggerFactory.getLogger(ModelQuantizer.class);
    public static final String QUANTIZATION_MANIFEST = "deliverance-quantization.json";
    private static final String README_NAME = "README.md";
    private static final String FINISHED_MARKER = ".finished";
    private static final int PROGRESS_LOG_INTERVAL = 25;
    private final long maxShardSize;

    public static final Predicate<String> DEFAULT_Q4_TENSOR_FILTER = name ->
            !name.endsWith(".qb")
                    && name.endsWith(".weight")
                    // Match the known-good external Q4 policy more closely: quantize only the large
                    // attention/MLP projection matrices, while keeping embeddings, norm vectors,
                    // lm_head, and miscellaneous dense weights in their original dtype.
                    && (
                    isAttentionProjection(name)
                            || isDenseMlpProjection(name)
                            || isQwenMoeExpertProjection(name)
                            || isMixtralMoeExpertProjection(name)
            );

    static boolean isAttentionProjection(String name) {
        return name.endsWith("self_attn.q_proj.weight")
                || name.endsWith("self_attn.k_proj.weight")
                || name.endsWith("self_attn.v_proj.weight")
                || name.endsWith("self_attn.o_proj.weight");
    }

    static boolean isDenseMlpProjection(String name) {
        return name.endsWith("mlp.gate_proj.weight")
                || name.endsWith("mlp.up_proj.weight")
                || name.endsWith("mlp.down_proj.weight");
    }

    static boolean isQwenMoeExpertProjection(String name) {
        return name.matches(".*\\.mlp\\.experts\\.\\d+\\.(gate_proj|up_proj|down_proj)\\.weight$");
    }

    static boolean isMixtralMoeExpertProjection(String name) {
        return name.matches(".*\\.block_sparse_moe\\.experts\\.\\d+\\.(w1|w2|w3)\\.weight$");
    }

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
        Instant startedAt = Instant.now();
        LOGGER.info("Quantizing model from {} to {} with target dtype {}", sourceDir, outputDir, targetType);
        try {
            Files.createDirectories(outputDir);
            LOGGER.info("Copying non-weight model files from {} to {}", sourceDir, outputDir);
            copyNonWeightFiles(sourceDir, outputDir);
            try (DefaultWeightLoader loader = new DefaultWeightLoader(sourceDir.toFile())) {
                List<TensorTransform> tensorTransforms = new ArrayList<>();
                List<String> tensorNames = tensorNamesToProcess(loader, sourceDir);
                LOGGER.info("Loaded {} tensors for quantization from {}", tensorNames.size(), sourceDir);
                Map<String, String> outputWeightMap = new LinkedHashMap<>();
                int[] shardCounter = new int[]{0};
                try {
                    for (int i = 0; i < tensorNames.size(); i++) {
                            String name = tensorNames.get(i);
                            AbstractTensor original = loader.load(name);
                            boolean quantized = false;
                            if (tensorFilter.test(name) && canQuantize(original, targetType)) {
                                LOGGER.info("Quantizing tensor {}/{} {} from {} to {}", i + 1, tensorNames.size(), name,
                                        original.dType(), targetType);
                                if (targetType == DType.Q4) {
                                    writeQ4Tensor(outputDir, loader.metadata(), outputWeightMap, shardCounter, name, original);
                                    quantized = true;
                                } else {
                                    try (AbstractTensor tensor = AbstractTensorUtils.quantize(original, targetType, true)) {
                                        writeTensor(outputDir, loader.metadata(), outputWeightMap, shardCounter, name, tensor);
                                        quantized = tensor != original;
                                    }
                                }
                            }
                            if (!quantized) {
                                writeTensor(outputDir, loader.metadata(), outputWeightMap, shardCounter, name, original);
                            }
                            tensorTransforms.add(TensorTransform.from(name, original, targetType, quantized));
                            original.close();
                            logProgress(i + 1, tensorNames.size(), startedAt);
                        }
                    LOGGER.info("Writing quantized model index to {}", outputDir);
                    JsonUtils.om.writeValue(outputDir.resolve(SafeTensorIndexPojo.MODEL_INDEX_JSON).toFile(),
                            new SafeTensorIndexPojo(loader.metadata(), outputWeightMap));
                    Files.deleteIfExists(outputDir.resolve(SafeTensorIndexPojo.SINGLE_MODEL_NAME));
                    writeQuantizationMetadata(sourceDir, outputDir, targetType, tensorTransforms);
                } finally {
                }
            }
            LOGGER.info("Finished quantizing model to {} in {} seconds", outputDir, Duration.between(startedAt, Instant.now()).toSeconds());
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

    boolean canQuantize(TensorInfo tensorInfo, DType targetType) {
        return tensorInfo.dType != targetType && tensorInfo.shape.length == 2;
    }

    /**
     * DefaultWeightLoader exposes both logical tensor names and internal `-part-*` chunks for
     * oversized tensors. The logical parent is not directly loadable, so the quantizer must skip
     * it and process the concrete parts instead.
     */
    boolean isLogicalSplitTensor(String name, DefaultWeightLoader loader) {
        return !name.contains("-part-") && loader.tensorInfoMap().containsKey(name + "-part-0");
    }

    private List<String> tensorNamesToProcess(DefaultWeightLoader loader, Path sourceDir) {
        List<String> names = new ArrayList<>();
        for (String name : loader.tensorInfoMap().keySet()) {
            if (name.endsWith(".qb") || isLogicalSplitTensor(name, loader)) {
                continue;
            }
            names.add(name);
        }
        Map<String, String> shardMap = sourceShardMap(sourceDir);
        names.sort(Comparator
                .comparing((String name) -> shardMap.getOrDefault(name, ""))
                .thenComparing(name -> name));
        return names;
    }

    private Map<String, String> sourceShardMap(Path sourceDir) {
        Path index = sourceDir.resolve(SafeTensorIndexPojo.MODEL_INDEX_JSON);
        if (!Files.exists(index)) {
            return Map.of();
        }
        try {
            return JsonUtils.om.readValue(index.toFile(), SafeTensorIndexPojo.class).getWeightFileMap();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private void writeTensor(Path outputDir, Map<String, String> metadata, Map<String, String> outputWeightMap,
            int[] shardCounter, String name, AbstractTensor tensor) {
        shardCounter[0]++;
        String shardName = String.format("model-%05d.safetensors", shardCounter[0]);
        List<SafeTensorWriter.NamedTensorPayload> payloads = SafeTensorWriter.flatten(Map.of(name, tensor));
        LOGGER.debug("Writing tensor {} to shard {} with {} payloads", name, shardName, payloads.size());
        SafeTensorWriter.writeShardFile(outputDir.resolve(shardName), metadata, payloads);
        for (SafeTensorWriter.NamedTensorPayload payload : payloads) {
            outputWeightMap.put(payload.name(), shardName);
        }
    }

    private void writeQ4Tensor(Path outputDir, Map<String, String> metadata, Map<String, String> outputWeightMap,
            int[] shardCounter, String name, AbstractTensor source) {
        if (source.size() % BLOCK_SIZE != 0) {
            throw new IllegalArgumentException("Q4 streaming requires tensor size to be a multiple of " + BLOCK_SIZE + ": " + name);
        }
        shardCounter[0]++;
        String shardName = String.format("model-%05d.safetensors", shardCounter[0]);
        Path outputFile = outputDir.resolve(shardName);
        try {
            Files.createDirectories(outputDir);
            long q4Bytes = source.size() / 2;
            long scaleBytes = (source.size() / BLOCK_SIZE) * Float.BYTES;
            Map<String, Object> header = new LinkedHashMap<>();
            if (metadata != null && !metadata.isEmpty()) {
                header.put("__metadata__", metadata);
            }
            header.put(name, tensorInfoMap(DType.Q4, source.shape().shapeArray(), 0, q4Bytes));
            header.put(name + ".qb", tensorInfoMap(DType.F32, q4BlockShape(source.shape().shapeArray()), q4Bytes, q4Bytes + scaleBytes));
            byte[] headerBytes = JsonUtils.om.writeValueAsBytes(header);
            try (FileChannel channel = FileChannel.open(outputFile,
                    StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE)) {
                ByteBuffer prefix = ByteBuffer.allocate(Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);
                prefix.putLong(headerBytes.length);
                prefix.flip();
                channel.write(prefix);
                channel.write(ByteBuffer.wrap(headerBytes));
                long dataStart = Long.BYTES + headerBytes.length;
                streamQ4Payloads(channel, dataStart, q4Bytes, source);
            }
            outputWeightMap.put(name, shardName);
            outputWeightMap.put(name + ".qb", shardName);
        } catch (IOException e) {
            throw new RuntimeException("Unable to write streaming Q4 tensor " + name + " to " + outputFile, e);
        }
    }

    private Map<String, Object> tensorInfoMap(DType dtype, int[] shape, long start, long end) {
        Map<String, Object> info = new LinkedHashMap<>();
        info.put("dtype", dtype.name());
        info.put("shape", SafeTensorWriter.canonicalShape(shape));
        info.put("data_offsets", new long[]{start, end});
        return info;
    }

    private int[] q4BlockShape(int[] sourceShape) {
        int[] shape = SafeTensorWriter.canonicalShape(sourceShape).clone();
        shape[shape.length - 1] = shape[shape.length - 1] / BLOCK_SIZE;
        return shape;
    }

    private void streamQ4Payloads(FileChannel channel, long dataStart, long scaleOffset, AbstractTensor source) throws IOException {
        ByteBuffer packed = ByteBuffer.allocate(HALF_BLOCK).order(ByteOrder.LITTLE_ENDIAN);
        ByteBuffer scaleBuffer = ByteBuffer.allocate(Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        long blocks = source.size() / BLOCK_SIZE;
        int columns = source.shape().last();
        for (long block = 0; block < blocks; block++) {
            int blockStart = Math.toIntExact(block * BLOCK_SIZE);
            float max = Float.MIN_VALUE;
            float absMax = Float.MIN_VALUE;
            for (int i = 0; i < BLOCK_SIZE; i++) {
                float value = sourceValue(source, blockStart + i, columns);
                float abs = value < 0 ? -value : value;
                if (abs > absMax) {
                    max = value;
                    absMax = abs;
                }
            }
            float scale = max / -8.0f;
            float inverseScale = scale != 0.0f ? 1.0f / scale : 0.0f;
            packed.clear();
            for (int i = 0; i < HALF_BLOCK; i++) {
                float f0 = sourceValue(source, blockStart + i, columns) * inverseScale;
                float f1 = sourceValue(source, blockStart + i + HALF_BLOCK, columns) * inverseScale;
                byte q0 = (byte) Math.min(15, (byte) (f0 + 8.5f));
                byte q1 = (byte) Math.min(15, (byte) (f1 + 8.5f));
                packed.put((byte) (q0 | (q1 << 4)));
            }
            packed.flip();
            channel.write(packed, dataStart + block * HALF_BLOCK);
            scaleBuffer.clear();
            scaleBuffer.putFloat(scale);
            scaleBuffer.flip();
            channel.write(scaleBuffer, dataStart + scaleOffset + block * Float.BYTES);
        }
    }

    private float sourceValue(AbstractTensor source, int flatOffset, int columns) {
        int row = flatOffset / columns;
        int column = flatOffset % columns;
        return source.get(row, column);
    }

    private void logProgress(int processed, int total, Instant startedAt) {
        if (processed == total || processed % PROGRESS_LOG_INTERVAL == 0) {
            LOGGER.info("Quantization progress: {}/{} tensors processed in {} seconds", processed, total,
                    Duration.between(startedAt, Instant.now()).toSeconds());
        }
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

    private void writeQuantizationMetadata(Path sourceDir, Path outputDir, DType targetType,
            List<TensorTransform> tensorTransforms) throws IOException {
        writeQuantizationReadme(sourceDir, outputDir, targetType, tensorTransforms);
        QuantizationManifest manifest = new QuantizationManifest(
                1,
                Instant.now().toString(),
                sourceDir.toAbsolutePath().toString(),
                outputDir.toAbsolutePath().toString(),
                targetType.name(),
                directorySize(sourceDir),
                directorySize(outputDir),
                List.copyOf(tensorTransforms));
        JsonUtils.om.writerWithDefaultPrettyPrinter().writeValue(outputDir.resolve(QUANTIZATION_MANIFEST).toFile(), manifest);
        Files.deleteIfExists(outputDir.resolve(FINISHED_MARKER));
        Files.createFile(outputDir.resolve(FINISHED_MARKER));
    }

    private void writeQuantizationReadme(Path sourceDir, Path outputDir, DType targetType,
            List<TensorTransform> tensorTransforms) throws IOException {
        Path sourceReadme = sourceDir.resolve(README_NAME);
        String original = Files.exists(sourceReadme) ? Files.readString(sourceReadme) : "";
        long quantizedCount = tensorTransforms.stream().filter(TensorTransform::quantized).count();
        long sourceSize = directorySize(sourceDir);
        long outputSize = directorySize(outputDir);
        String section = "# Deliverance Quantization\n\n"
                + "This model directory was generated by **Deliverance Q.O.D. (Quantize On Demand)** from a local source model.\n\n"
                + "Deliverance rewrote selected matrix weights into its `" + targetType.name() + "` quantized tensor format "
                + "to reduce model size and improve local inference throughput while keeping the model card and tokenizer metadata intact.\n\n"
                + "## What Changed\n\n"
                + "- Original local size: `" + humanReadableBytes(sourceSize) + "`\n"
                + "- Quantized local size: `" + humanReadableBytes(outputSize) + "`\n"
                + "- Target dtype: `" + targetType.name() + "`\n"
                + "- Quantized tensors: `" + quantizedCount + "`\n"
                + "- Provenance manifest: `" + QUANTIZATION_MANIFEST + "`\n\n"
                + "The manifest records the source/output paths, tensor dtype changes, generated `.qb` sidecars, "
                + "and shape-normalization transforms used during export.\n\n"
                + "## Try Deliverance\n\n"
                + "Deliverance is a local Java inference engine with safetensors loading, Q4 quantization, "
                + "and quantize-on-demand model generation. Learn more at "
                + "[github.com/edwardcapriolo/deliverance](https://github.com/edwardcapriolo/deliverance).\n\n";
        if (original.isBlank()) {
            Files.writeString(outputDir.resolve(README_NAME), section);
        } else {
            Files.writeString(outputDir.resolve(README_NAME), insertAfterModelCardHeader(original, section));
        }
    }

    private String insertAfterModelCardHeader(String original, String section) {
        int insertAt = modelCardHeaderEnd(original);
        return original.substring(0, insertAt)
                + "\n\n---\n\n"
                + section
                + original.substring(insertAt).stripLeading();
    }

    private int modelCardHeaderEnd(String original) {
        int cursor = 0;
        if (original.startsWith("---\n")) {
            int frontMatterEnd = original.indexOf("\n---", 4);
            if (frontMatterEnd >= 0) {
                int afterClosingLine = original.indexOf('\n', frontMatterEnd + 1);
                cursor = afterClosingLine < 0 ? original.length() : afterClosingLine + 1;
            }
        }
        int headingStart = firstHeadingAtOrAfter(original, cursor);
        if (headingStart >= 0) {
            int nextBlankLine = original.indexOf("\n\n", headingStart);
            return nextBlankLine < 0 ? original.length() : nextBlankLine + 2;
        }
        if (cursor > 0) {
            return cursor;
        }
        int nextBlankLine = original.indexOf("\n\n");
        return nextBlankLine < 0 ? original.length() : nextBlankLine + 2;
    }

    private int firstHeadingAtOrAfter(String original, int start) {
        int cursor = Math.max(0, start);
        while (cursor < original.length()) {
            int lineEnd = original.indexOf('\n', cursor);
            int end = lineEnd < 0 ? original.length() : lineEnd;
            if (end > cursor && original.charAt(cursor) == '#') {
                return cursor;
            }
            if (lineEnd < 0) {
                return -1;
            }
            cursor = lineEnd + 1;
        }
        return -1;
    }

    private long directorySize(Path directory) throws IOException {
        if (!Files.exists(directory)) {
            return 0;
        }
        try (Stream<Path> paths = Files.walk(directory)) {
            return paths.filter(Files::isRegularFile)
                    .mapToLong(path -> {
                        try {
                            return Files.size(path);
                        } catch (IOException e) {
                            throw new RuntimeException("Unable to size " + path, e);
                        }
                    })
                    .sum();
        }
    }

    private String humanReadableBytes(long bytes) {
        if (bytes < 1024) {
            return bytes + " B";
        }
        String[] units = {"KB", "MB", "GB", "TB"};
        double value = bytes;
        int unit = -1;
        do {
            value /= 1024.0;
            unit++;
        } while (value >= 1024.0 && unit < units.length - 1);
        return String.format("%.1f %s", value, units[unit]);
    }

    public record QuantizationManifest(
            int schemaVersion,
            String createdAt,
            String sourceDirectory,
            String outputDirectory,
            String targetType,
            long sourceSizeBytes,
            long outputSizeBytes,
            List<TensorTransform> tensorTransforms) {
    }

    public record TensorTransform(
            String name,
            String sourceDType,
            String outputDType,
            int[] sourceShape,
            int[] outputShape,
            boolean quantized,
            List<String> sidecars,
            boolean oneDimensionalToRowVector) {

        static TensorTransform from(String name, AbstractTensor source, AbstractTensor output, boolean quantized) {
            int[] sourceShape = source.shape().shapeArray();
            int[] outputShape = SafeTensorWriter.canonicalShape(output.shape().shapeArray());
            return new TensorTransform(
                    name,
                    source.dType().name(),
                    output.dType().name(),
                    sourceShape,
                    outputShape,
                    quantized,
                    sidecarsFor(name, output.dType()),
                    sourceShape.length == 1 && outputShape.length == 2);
        }

        static TensorTransform from(String name, AbstractTensor source, DType outputType, boolean quantized) {
            int[] sourceShape = source.shape().shapeArray();
            int[] outputShape = SafeTensorWriter.canonicalShape(source.shape().shapeArray());
            DType dtype = quantized ? outputType : source.dType();
            return new TensorTransform(
                    name,
                    source.dType().name(),
                    dtype.name(),
                    sourceShape,
                    outputShape,
                    quantized,
                    sidecarsFor(name, dtype),
                    sourceShape.length == 1 && outputShape.length == 2);
        }

        private static List<String> sidecarsFor(String name, DType dType) {
            if (dType == DType.Q4 || dType == DType.I8) {
                return List.of(name + ".qb");
            }
            return List.of();
        }
    }
}
