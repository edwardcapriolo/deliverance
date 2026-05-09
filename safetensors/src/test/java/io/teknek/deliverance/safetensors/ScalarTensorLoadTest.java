package io.teknek.deliverance.safetensors;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class ScalarTensorLoadTest {
    @TempDir
    Path tempDir;

    @Test
    public void loadsZeroDimensionalScalarTensor() throws Exception {
        Path output = tempDir.resolve("model.safetensors");
        byte[] header = """
                {"scalar":{"dtype":"F32","shape":[],"data_offsets":[0,4]}}
                """.getBytes(StandardCharsets.UTF_8);

        try (FileChannel channel = FileChannel.open(output,
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE)) {
            ByteBuffer prefix = ByteBuffer.allocate(Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);
            prefix.putLong(header.length);
            prefix.flip();
            channel.write(prefix);
            channel.write(ByteBuffer.wrap(header));

            ByteBuffer payload = ByteBuffer.allocate(Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
            payload.putFloat(42.5f);
            payload.flip();
            channel.write(payload);
        }

        try (DefaultWeightLoader loader = new DefaultWeightLoader(tempDir.toFile());
             var scalar = loader.load("scalar")) {
            assertEquals(1, scalar.shape().first());
            assertEquals(1, scalar.shape().last());
            assertEquals(42.5f, scalar.get(0, 0), 1.0e-6f);
        }
    }
}
