package io.teknek.deliverance.safetensors;

import io.teknek.deliverance.tensor.TensorInfo;
import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class DefaultWeightLoaderHeaderTest {

    @Test
    void readsMetadataAndOrdersTensorsByDataOffset() {
        String json = "{\"z\":{\"dtype\":\"F32\",\"shape\":[1,1],\"data_offsets\":[4,8]},"
                + "\"__METADATA__\":{\"format\":\"pt\"},"
                + "\"a\":{\"dtype\":\"F32\",\"shape\":[1,1],\"data_offsets\":[0,4]}}";
        Map<String, String> metadata = new HashMap<>();

        Map<String, TensorInfo> infos = DefaultWeightLoader.readTensorInfoMap(
                header(json), Optional.of(metadata));

        assertEquals(java.util.List.of("a", "z"), infos.keySet().stream().toList());
        assertEquals(Map.of("format", "pt"), metadata);
    }

    @Test
    void rejectsNegativeHeaderLength() {
        ByteBuffer buffer = ByteBuffer.allocate(Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        buffer.putLong(-1L).flip();

        assertThrows(IllegalArgumentException.class,
                () -> DefaultWeightLoader.readTensorInfoMap(buffer, Optional.empty()));
    }

    @Test
    void rejectsHeaderLargerThanConfiguredLimitBeforeAllocation() {
        ByteBuffer buffer = ByteBuffer.allocate(Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        buffer.putLong(1024L * 1024L * 1024L + 1L).flip();

        assertThrows(IllegalArgumentException.class,
                () -> DefaultWeightLoader.readTensorInfoMap(buffer, Optional.empty()));
    }

    @Test
    void rejectsTruncatedHeaderPayload() {
        ByteBuffer buffer = ByteBuffer.allocate(Long.BYTES + 2).order(ByteOrder.LITTLE_ENDIAN);
        buffer.putLong(3L).put((byte) '{').put((byte) '}').flip();

        assertThrows(java.nio.BufferUnderflowException.class,
                () -> DefaultWeightLoader.readTensorInfoMap(buffer, Optional.empty()));
    }

    @Test
    void rejectsMalformedJsonHeader() {
        assertThrows(RuntimeException.class,
                () -> DefaultWeightLoader.readTensorInfoMap(header("{not-json}"), Optional.empty()));
    }

    private static ByteBuffer header(String json) {
        byte[] bytes = json.getBytes(StandardCharsets.UTF_8);
        ByteBuffer buffer = ByteBuffer.allocate(Long.BYTES + bytes.length).order(ByteOrder.LITTLE_ENDIAN);
        buffer.putLong(bytes.length).put(bytes).flip();
        return buffer;
    }
}
