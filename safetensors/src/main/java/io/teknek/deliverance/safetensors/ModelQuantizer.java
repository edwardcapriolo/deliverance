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
    private static final int Q4_WRITE_BUFFER_BYTES = 1 << 20;
    private final long maxShardSize;
    private final ReadMode readMode;

    public enum ReadMode {
        DEFAULT_WEIGHT_LOADER,
        SHARD_WEIGHT_LOADER
    }

    public static final Predicate<String> DEFAULT_Q4_TENSOR_FILTER = name ->
            !name.endsWith(".qb")
                    && name.endsWith(".weight")
                    // Match the known-good external Q4 policy more closely: quantize only the large
                    // attention/MLP projection matrices, while keeping embeddings, norm vectors,
                    // lm_head, and miscellaneous dense weights in their original dtype.
                    && (
                    isAttentionProjection(name)
                            || isDenseMlpProjection(name)
                            || isGraniteMoeHybridProjection(name)
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

    static boolean isGraniteMoeHybridProjection(String name) {
        return name.endsWith("shared_mlp.input_linear.weight")
                || name.endsWith("shared_mlp.output_linear.weight")
                || name.endsWith("block_sparse_moe.input_linear.weight")
                || name.endsWith("block_sparse_moe.output_linear.weight")
                || name.endsWith("mamba.in_proj.weight")
                || name.endsWith("mamba.out_proj.weight");
    }

    static boolean isMixtralMoeExpertProjection(String name) {
        return name.matches(".*\\.block_sparse_moe\\.experts\\.\\d+\\.(w1|w2|w3)\\.weight$");
    }

    public ModelQuantizer() {
        this(SafeTensorWriter.DEFAULT_MAX_SHARD_SIZE);
    }

    public ModelQuantizer(long maxShardSize) {
        this(maxShardSize, ReadMode.SHARD_WEIGHT_LOADER);
    }

    public ModelQuantizer(long maxShardSize, ReadMode readMode) {
        this.maxShardSize = maxShardSize;
        this.readMode = readMode;
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
            QuantizationRun run = readMode == ReadMode.SHARD_WEIGHT_LOADER
                    ? quantizeWithShardLoader(sourceDir, outputDir, targetType, tensorFilter, startedAt)
                    : quantizeWithDefaultLoader(sourceDir, outputDir, targetType, tensorFilter, startedAt);
            LOGGER.info("Writing quantized model index to {}", outputDir);
            JsonUtils.om.writeValue(outputDir.resolve(SafeTensorIndexPojo.MODEL_INDEX_JSON).toFile(),
                    new SafeTensorIndexPojo(run.metadata(), run.outputWeightMap()));
            Files.deleteIfExists(outputDir.resolve(SafeTensorIndexPojo.SINGLE_MODEL_NAME));
            writeQuantizationMetadata(sourceDir, outputDir, targetType, run.tensorTransforms());
            LOGGER.info("Finished quantizing model to {} in {} seconds", outputDir, Duration.between(startedAt, Instant.now()).toSeconds());
        } catch (IOException e) {
            throw new RuntimeException("Unable to quantize model from " + sourceDir + " to " + outputDir, e);
        }
    }

    private QuantizationRun quantizeWithDefaultLoader(Path sourceDir, Path outputDir, DType targetType,
            Predicate<String> tensorFilter, Instant startedAt) {
        try (DefaultWeightLoader loader = new DefaultWeightLoader(sourceDir.toFile())) {
            List<String> tensorNames = tensorNamesToProcess(loader, sourceDir);
            LOGGER.info("Loaded {} tensors for quantization from {} using default loader", tensorNames.size(), sourceDir);
            return quantizeTensorList(loader, tensorNames, outputDir, targetType, tensorFilter, startedAt);
        }
    }

    private QuantizationRun quantizeWithShardLoader(Path sourceDir, Path outputDir, DType targetType,
            Predicate<String> tensorFilter, Instant startedAt) {
        Map<String, String> sourceShardMap = sourceShardMap(sourceDir);
        if (sourceShardMap.isEmpty()) {
            Path single = sourceDir.resolve(SafeTensorIndexPojo.SINGLE_MODEL_NAME);
            try (SafetensorsShardWeightLoader loader = new SafetensorsShardWeightLoader(single)) {
                return quantizeTensorList(loader, tensorNamesToProcess(loader, sourceDir), outputDir,
                        targetType, tensorFilter, startedAt);
            }
        }

        Map<String, List<String>> byShard = new LinkedHashMap<>();
        for (Map.Entry<String, String> entry : sourceShardMap.entrySet()) {
            if (!entry.getKey().endsWith(".qb")) {
                byShard.computeIfAbsent(entry.getValue(), ignored -> new ArrayList<>()).add(entry.getKey());
            }
        }
        List<TensorTransform> transforms = new ArrayList<>();
        Map<String, String> outputWeightMap = new LinkedHashMap<>();
        Map<String, String> metadata = Map.of();
        int[] shardCounter = new int[]{0};
        int[] processed = new int[]{0};
        int total = byShard.values().stream().mapToInt(List::size).sum();
        for (Map.Entry<String, List<String>> shard : byShard.entrySet()) {
            LOGGER.info("Opening source shard {} with {} tensors", shard.getKey(), shard.getValue().size());
            try (SafetensorsShardWeightLoader loader = new SafetensorsShardWeightLoader(sourceDir.resolve(shard.getKey()))) {
                metadata = loader.metadata();
                List<String> names = shard.getValue().stream().sorted().toList();
                quantizeNames(loader, names, outputDir, targetType, tensorFilter, startedAt,
                        outputWeightMap, transforms, shardCounter, processed, total);
            }
        }
        return new QuantizationRun(metadata, outputWeightMap, transforms);
    }

    private QuantizationRun quantizeTensorList(WeightLoader loader, List<String> tensorNames, Path outputDir,
            DType targetType, Predicate<String> tensorFilter, Instant startedAt) {
        List<TensorTransform> tensorTransforms = new ArrayList<>();
        Map<String, String> outputWeightMap = new LinkedHashMap<>();
        int[] shardCounter = new int[]{0};
        int[] processed = new int[]{0};
        quantizeNames(loader, tensorNames, outputDir, targetType, tensorFilter, startedAt,
                outputWeightMap, tensorTransforms, shardCounter, processed, tensorNames.size());
        return new QuantizationRun(loader.metadata(), outputWeightMap, tensorTransforms);
    }

    private void quantizeNames(WeightLoader loader, List<String> tensorNames, Path outputDir, DType targetType,
            Predicate<String> tensorFilter, Instant startedAt, Map<String, String> outputWeightMap,
            List<TensorTransform> tensorTransforms, int[] shardCounter, int[] processed, int total) {
        if (targetType == DType.Q4) {
            quantizeNamesToPackedQ4Shards(loader, tensorNames, outputDir, tensorFilter, startedAt,
                    outputWeightMap, tensorTransforms, shardCounter, processed, total);
            return;
        }
        for (String name : tensorNames) {
            processed[0]++;
            AbstractTensor original = loader.load(name);
            boolean quantized = false;
            if (tensorFilter.test(name) && canQuantize(original, targetType)) {
                LOGGER.info("Quantizing tensor {}/{} {} from {} to {}", processed[0], total, name,
                        original.dType(), targetType);
                try (AbstractTensor tensor = AbstractTensorUtils.quantize(original, targetType, true)) {
                    writeTensor(outputDir, loader.metadata(), outputWeightMap, shardCounter, name, tensor);
                    quantized = tensor != original;
                }
            }
            if (!quantized) {
                writeTensor(outputDir, loader.metadata(), outputWeightMap, shardCounter, name, original);
            }
            tensorTransforms.add(TensorTransform.from(name, original, targetType, quantized));
            original.close();
            logProgress(processed[0], total, startedAt);
        }
    }

    private void quantizeNamesToPackedQ4Shards(WeightLoader loader, List<String> tensorNames, Path outputDir,
            Predicate<String> tensorFilter, Instant startedAt, Map<String, String> outputWeightMap,
            List<TensorTransform> tensorTransforms, int[] shardCounter, int[] processed, int total) {
        List<OutputTensorPlan> currentShard = new ArrayList<>();
        long currentShardBytes = 0;
        for (String name : tensorNames) {
            OutputTensorPlan plan = outputTensorPlan(name, loader.tensorInfoMap().get(name), tensorFilter);
            long tensorBytes = plan.length();
            if (!currentShard.isEmpty() && currentShardBytes + tensorBytes > maxShardSize) {
                writePackedQ4Shard(outputDir, loader, currentShard, outputWeightMap, tensorTransforms,
                        shardCounter, processed, total, startedAt);
                currentShard.clear();
                currentShardBytes = 0;
            }
            currentShard.add(plan);
            currentShardBytes += tensorBytes;
        }
        if (!currentShard.isEmpty()) {
            writePackedQ4Shard(outputDir, loader, currentShard, outputWeightMap, tensorTransforms,
                    shardCounter, processed, total, startedAt);
        }
    }

    private OutputTensorPlan outputTensorPlan(String name, TensorInfo info, Predicate<String> tensorFilter) {
        if (info == null) {
            throw new IllegalArgumentException("Missing tensor info for " + name);
        }
        boolean quantized = tensorFilter.test(name) && canQuantize(info, DType.Q4);
        if (quantized) {
            long elements = elementCount(info.shape);
            if (elements % BLOCK_SIZE != 0) {
                throw new IllegalArgumentException("Q4 streaming requires tensor size to be a multiple of "
                        + BLOCK_SIZE + ": " + name);
            }
            long q4Bytes = elements / 2;
            long scaleBytes = (elements / BLOCK_SIZE) * Float.BYTES;
            return new OutputTensorPlan(name, quantized, List.of(
                    new OutputPayloadPlan(name, DType.Q4, info.shape, q4Bytes),
                    new OutputPayloadPlan(name + ".qb", DType.F32, q4BlockShape(info.shape), scaleBytes)));
        }
        return new OutputTensorPlan(name, false, List.of(
                new OutputPayloadPlan(name, info.dType, info.shape, info.dataOffsets[1] - info.dataOffsets[0])));
    }

    private void writePackedQ4Shard(Path outputDir, WeightLoader loader, List<OutputTensorPlan> tensors,
            Map<String, String> outputWeightMap, List<TensorTransform> tensorTransforms, int[] shardCounter,
            int[] processed, int total, Instant startedAt) {
        shardCounter[0]++;
        String shardName = String.format("model-%05d.safetensors", shardCounter[0]);
        Path outputFile = outputDir.resolve(shardName);
        try {
            Files.createDirectories(outputDir);
            Map<String, Object> header = new LinkedHashMap<>();
            Map<String, Long> payloadOffsets = new LinkedHashMap<>();
            Map<String, Long> payloadEnds = new LinkedHashMap<>();
            Map<String, String> metadata = loader.metadata();
            if (metadata != null && !metadata.isEmpty()) {
                header.put("__metadata__", metadata);
            }
            long offset = 0;
            for (OutputTensorPlan tensor : tensors) {
                for (OutputPayloadPlan payload : tensor.payloads()) {
                    long end = offset + payload.length();
                    header.put(payload.name(), tensorInfoMap(payload.dtype(), payload.shape(), offset, end));
                    payloadOffsets.put(payload.name(), offset);
                    payloadEnds.put(payload.name(), end);
                    outputWeightMap.put(payload.name(), shardName);
                    offset = end;
                }
            }
            byte[] headerBytes = JsonUtils.om.writeValueAsBytes(header);
            LOGGER.debug("Writing packed quantized shard {} with {} tensors and {} payload bytes",
                    shardName, tensors.size(), offset);
            try (FileChannel channel = FileChannel.open(outputFile,
                    StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE)) {
                ByteBuffer prefix = ByteBuffer.allocate(Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);
                prefix.putLong(headerBytes.length);
                prefix.flip();
                channel.write(prefix);
                channel.write(ByteBuffer.wrap(headerBytes));
                long dataStart = Long.BYTES + headerBytes.length;
                for (OutputTensorPlan tensor : tensors) {
                    processed[0]++;
                    try (AbstractTensor original = loader.load(tensor.name())) {
                        if (tensor.quantized()) {
                            LOGGER.info("Quantizing tensor {}/{} {} from {} to {}", processed[0], total,
                                    tensor.name(), original.dType(), DType.Q4);
                            String q4Name = tensor.name();
                            String scaleName = tensor.name() + ".qb";
                            streamQ4Payloads(channel,
                                    dataStart + payloadOffsets.get(q4Name),
                                    dataStart + payloadOffsets.get(scaleName),
                                    original);
                        } else {
                            writeDensePayload(channel, dataStart + payloadOffsets.get(tensor.name()), original);
                        }
                        tensorTransforms.add(TensorTransform.from(tensor.name(), original, DType.Q4, tensor.quantized()));
                    }
                    logProgress(processed[0], total, startedAt);
                }
            }
            for (Map.Entry<String, Long> entry : payloadEnds.entrySet()) {
                if (entry.getValue() < payloadOffsets.get(entry.getKey())) {
                    throw new IllegalStateException("Invalid payload offsets for " + entry.getKey());
                }
            }
        } catch (IOException e) {
            throw new RuntimeException("Unable to write packed Q4 shard " + outputFile, e);
        }
    }

    private long elementCount(int[] shape) {
        long elements = 1;
        for (long dimension : shape) {
            elements = Math.multiplyExact(elements, dimension);
        }
        return elements;
    }

    private void writeDensePayload(FileChannel channel, long position, AbstractTensor tensor) throws IOException {
        ByteBuffer bytes = tensor.getMemorySegment().asByteBuffer().duplicate().order(ByteOrder.LITTLE_ENDIAN);
        bytes.clear();
        while (bytes.hasRemaining()) {
            channel.write(bytes, position + bytes.position());
        }
    }

    private record OutputTensorPlan(String name, boolean quantized, List<OutputPayloadPlan> payloads) {
        long length() {
            return payloads.stream().mapToLong(OutputPayloadPlan::length).sum();
        }
    }

    private record OutputPayloadPlan(String name, DType dtype, int[] shape, long length) {
    }

    private record QuantizationRun(Map<String, String> metadata, Map<String, String> outputWeightMap,
            List<TensorTransform> tensorTransforms) {
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
    boolean isLogicalSplitTensor(String name, WeightLoader loader) {
        return !name.contains("-part-") && loader.tensorInfoMap().containsKey(name + "-part-0");
    }

    private List<String> tensorNamesToProcess(WeightLoader loader, Path sourceDir) {
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

    private void streamQ4Payloads(FileChannel channel, long q4Start, long scaleStart, AbstractTensor source) throws IOException {
        int packedCapacity = Math.max(HALF_BLOCK, Q4_WRITE_BUFFER_BYTES - (Q4_WRITE_BUFFER_BYTES % HALF_BLOCK));
        int scaleCapacity = (packedCapacity / HALF_BLOCK) * Float.BYTES;
        ByteBuffer packed = ByteBuffer.allocateDirect(packedCapacity).order(ByteOrder.LITTLE_ENDIAN);
        ByteBuffer scaleBuffer = ByteBuffer.allocateDirect(scaleCapacity).order(ByteOrder.LITTLE_ENDIAN);
        long blocks = source.size() / BLOCK_SIZE;
        int columns = source.shape().last();
        long packedBytesWritten = 0;
        long scaleBytesWritten = 0;
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
            for (int i = 0; i < HALF_BLOCK; i++) {
                float f0 = sourceValue(source, blockStart + i, columns) * inverseScale;
                float f1 = sourceValue(source, blockStart + i + HALF_BLOCK, columns) * inverseScale;
                byte q0 = (byte) Math.min(15, (byte) (f0 + 8.5f));
                byte q1 = (byte) Math.min(15, (byte) (f1 + 8.5f));
                packed.put((byte) (q0 | (q1 << 4)));
            }
            scaleBuffer.putFloat(scale);
            if (!packed.hasRemaining()) {
                packedBytesWritten += flush(channel, packed, q4Start + packedBytesWritten);
                scaleBytesWritten += flush(channel, scaleBuffer, scaleStart + scaleBytesWritten);
            }
        }
        if (packed.position() > 0) {
            packedBytesWritten += flush(channel, packed, q4Start + packedBytesWritten);
            scaleBytesWritten += flush(channel, scaleBuffer, scaleStart + scaleBytesWritten);
        }
    }

    private long flush(FileChannel channel, ByteBuffer buffer, long position) throws IOException {
        buffer.flip();
        int bytes = buffer.remaining();
        while (buffer.hasRemaining()) {
            channel.write(buffer, position + buffer.position());
        }
        buffer.clear();
        return bytes;
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
