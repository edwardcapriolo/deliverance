package io.teknek.deliverance.safetensors;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SafetensorsShardWeightLoaderTest {
    @TempDir
    Path tempDir;

    @Test
    void loadsMetadataAndTensorFromOneShard() {
        Path shard = writeModel();

        try (SafetensorsShardWeightLoader loader = new SafetensorsShardWeightLoader(shard);
             AbstractTensor tensor = loader.load("layer.weight")) {
            assertEquals(Map.of("format", "pt"), loader.metadata());
            assertEquals(DType.F32, loader.getModelDType());
            assertEquals(2, tensor.shape().first());
            assertEquals(3, tensor.shape().last());
            assertEquals(6.0f, tensor.get(1, 2));
        }
    }

    @Test
    void copiesRawPayloadWithoutHeaderBytes() throws Exception {
        Path shard = writeModel();
        byte[] fileBytes = Files.readAllBytes(shard);
        long headerLengthLong = ByteBuffer.wrap(fileBytes).order(ByteOrder.LITTLE_ENDIAN).getLong();
        assertTrue(headerLengthLong <= Integer.MAX_VALUE);
        int headerLength = (int) headerLengthLong;
        int dataStart = Long.BYTES + headerLength;

        Path copied = tempDir.resolve("copied.bin");
        try (SafetensorsShardWeightLoader loader = new SafetensorsShardWeightLoader(shard);
             FileChannel output = FileChannel.open(copied, StandardOpenOption.CREATE,
                     StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE)) {
            loader.copyRawPayload("layer.weight", output, 0);
        }

        long[] offsets;
        try (SafetensorsShardWeightLoader loader = new SafetensorsShardWeightLoader(shard)) {
            offsets = loader.tensorInfoMap().get("layer.weight").dataOffsets;
        }
        byte[] expected = java.util.Arrays.copyOfRange(fileBytes, dataStart + (int) offsets[0],
                dataStart + (int) offsets[1]);
        assertArrayEquals(expected, Files.readAllBytes(copied));
    }

    @Test
    void rejectsUnknownRawPayloadName() throws Exception {
        Path shard = writeModel();
        Path copied = tempDir.resolve("missing.bin");
        try (SafetensorsShardWeightLoader loader = new SafetensorsShardWeightLoader(shard);
             FileChannel output = FileChannel.open(copied, StandardOpenOption.CREATE,
                     StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE)) {
            assertThrows(IllegalArgumentException.class,
                    () -> loader.copyRawPayload("missing", output, 0));
        }
    }

    private Path writeModel() {
        FloatBufferTensor tensor = new FloatBufferTensor(2, 3);
        for (int row = 0; row < 2; row++) {
            for (int col = 0; col < 3; col++) {
                tensor.set(row * 3 + col + 1.0f, row, col);
            }
        }
        Path shard = tempDir.resolve("model-00001-of-00001.safetensors");
        SafeTensorWriter.write(shard, Map.of("format", "pt"), Map.of("layer.weight", tensor));
        tensor.close();
        return shard;
    }
}
