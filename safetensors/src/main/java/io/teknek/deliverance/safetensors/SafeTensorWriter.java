package io.teknek.deliverance.safetensors;

import com.fasterxml.jackson.core.JsonProcessingException;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorInfo;
import io.teknek.deliverance.tensor.impl.Q4ByteBufferTensor;
import io.teknek.deliverance.tensor.impl.Q8ByteBufferTensor;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class SafeTensorWriter {
    public static final long DEFAULT_MAX_SHARD_SIZE = 1_000_000_000L;
    private static final String METADATA_KEY = "__metadata__";

    private SafeTensorWriter() {
    }

    public static void write(Path outputFile, Map<String, String> metadata, Map<String, AbstractTensor> tensors) {
        writeShardFile(outputFile, metadata, flatten(tensors));
    }

    public static void writeModel(Path outputDir, Map<String, String> metadata, Map<String, AbstractTensor> tensors) {
        writeModel(outputDir, metadata, tensors, DEFAULT_MAX_SHARD_SIZE);
    }

    public static void writeModel(Path outputDir, Map<String, String> metadata, Map<String, AbstractTensor> tensors, long maxShardSize) {
        try {
            Files.createDirectories(outputDir);
            List<NamedTensorPayload> payloads = flatten(tensors);
            List<List<NamedTensorPayload>> shards = shard(payloads, maxShardSize);
            if (shards.size() == 1) {
                writeShardFile(outputDir.resolve(SafeTensorIndexPojo.SINGLE_MODEL_NAME), metadata, shards.get(0));
                Files.deleteIfExists(outputDir.resolve(SafeTensorIndexPojo.MODEL_INDEX_JSON));
                return;
            }

            Map<String, String> weightMap = new LinkedHashMap<>();
            for (int i = 0; i < shards.size(); i++) {
                String shardName = shardFileName(i, shards.size());
                List<NamedTensorPayload> shardPayloads = shards.get(i);
                writeShardFile(outputDir.resolve(shardName), metadata, shardPayloads);
                for (NamedTensorPayload payload : shardPayloads) {
                    weightMap.put(payload.name(), shardName);
                }
            }

            SafeTensorIndexPojo index = new SafeTensorIndexPojo(metadata == null ? Map.of() : metadata, weightMap);
            JsonUtils.om.writeValue(outputDir.resolve(SafeTensorIndexPojo.MODEL_INDEX_JSON).toFile(), index);
            Files.deleteIfExists(outputDir.resolve(SafeTensorIndexPojo.SINGLE_MODEL_NAME));
        } catch (IOException e) {
            throw new RuntimeException("Unable to write sharded safetensors model in " + outputDir, e);
        }
    }

    static void writeShardFile(Path outputFile, Map<String, String> metadata, List<NamedTensorPayload> payloads) {
        try {
            if (outputFile.getParent() != null) {
                Files.createDirectories(outputFile.getParent());
            }
            Serialized serialized = serialize(metadata, payloads);
            try (FileChannel channel = FileChannel.open(outputFile,
                    StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE)) {
                ByteBuffer prefix = ByteBuffer.allocate(Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);
                prefix.putLong(serialized.header.length);
                prefix.flip();
                channel.write(prefix);
                channel.write(ByteBuffer.wrap(serialized.header));
                for (ByteBuffer buffer : serialized.payloads.values()) {
                    channel.write(buffer.duplicate());
                }
            }
        } catch (IOException e) {
            throw new RuntimeException("Unable to write safetensors file " + outputFile, e);
        }
    }

    static Serialized serialize(Map<String, String> metadata, List<NamedTensorPayload> payloads) {
        try {
            Map<String, Object> header = new LinkedHashMap<>();
            if (metadata != null && !metadata.isEmpty()) {
                header.put(METADATA_KEY, metadata);
            }

            Map<String, ByteBuffer> payloadBytes = new LinkedHashMap<>();
            long offset = 0;
            for (NamedTensorPayload payload : payloads) {
                header.put(payload.name(), toTensorInfo(payload.dType(), payload.shape(), offset, offset + payload.length()));
                payloadBytes.put(payload.name(), payload.bytes());
                offset += payload.length();
            }

            return new Serialized(JsonUtils.om.writeValueAsBytes(header), payloadBytes);
        } catch (JsonProcessingException e) {
            throw new RuntimeException("Unable to serialize safetensors header", e);
        }
    }

    static List<NamedTensorPayload> flatten(Map<String, AbstractTensor> tensors) {
        List<NamedTensorPayload> payloads = new ArrayList<>();
        for (Map.Entry<String, AbstractTensor> entry : tensors.entrySet()) {
            String name = entry.getKey();
            AbstractTensor tensor = entry.getValue();
            payloads.add(named(name, tensor));
            if (tensor instanceof Q4ByteBufferTensor q4) {
                payloads.add(named(name + ".qb", q4.getBlockF()));
            } else if (tensor instanceof Q8ByteBufferTensor q8) {
                payloads.add(named(name + ".qb", q8.getBlockF()));
            }
        }
        return payloads;
    }

    static List<List<NamedTensorPayload>> shard(List<NamedTensorPayload> payloads, long maxShardSize) {
        List<List<NamedTensorPayload>> shards = new ArrayList<>();
        List<NamedTensorPayload> current = new ArrayList<>();
        long currentSize = 0;
        for (NamedTensorPayload payload : payloads) {
            if (!current.isEmpty() && currentSize + payload.length() > maxShardSize) {
                shards.add(current);
                current = new ArrayList<>();
                currentSize = 0;
            }
            current.add(payload);
            currentSize += payload.length();
        }
        if (!current.isEmpty()) {
            shards.add(current);
        }
        return shards;
    }

    private static NamedTensorPayload named(String name, AbstractTensor tensor) {
        ByteBuffer bytes = tensor.getMemorySegment().asByteBuffer().duplicate().order(ByteOrder.LITTLE_ENDIAN);
        bytes.clear();
        return new NamedTensorPayload(name, tensor.dType(), tensor.shape().shapeArray(), bytes.slice().order(ByteOrder.LITTLE_ENDIAN));
    }

    private static TensorInfo toTensorInfo(DType dType, int[] shape, long start, long end) {
        long[] longShape = new long[shape.length];
        for (int i = 0; i < shape.length; i++) {
            longShape[i] = shape[i];
        }
        return new TensorInfo(dType, longShape, new long[]{start, end});
    }

    private static String shardFileName(int index, int total) {
        return String.format("model-%05d-of-%05d.safetensors", index + 1, total);
    }

    record Serialized(byte[] header, Map<String, ByteBuffer> payloads) {
    }

    record NamedTensorPayload(String name, DType dType, int[] shape, ByteBuffer bytes) {
        int length() {
            return bytes.remaining();
        }
    }
}
