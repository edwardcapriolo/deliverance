package io.teknek.deliverance.safetensors;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class LoraAdapterConfigTest {

    @TempDir
    Path tempDir;

    /**
     * Fixture mirrors the real shape of a published adapter_config.json
     * (bunnycore/Llama-3.2-1b-chatml-lora_model), including the many fields Deliverance
     * doesn't need -- confirms {@code @JsonIgnoreProperties(ignoreUnknown = true)} works
     * against real-world PEFT output, not just a minimal hand-crafted JSON object.
     */
    @Test
    void parsesRealShapeConfigIgnoringUnknownFields() throws IOException {
        LoraAdapterConfig config = JsonUtils.om.readValue(fixtureBytes(), LoraAdapterConfig.class);

        assertEquals(16, config.rank);
        assertEquals(16.0, config.alpha);
        assertEquals(1.0, config.scale());
        assertEquals(
                List.of("k_proj", "v_proj", "o_proj", "down_proj", "q_proj", "up_proj", "gate_proj"),
                config.targetModules);
    }

    @Test
    void loadReadsAdapterConfigJsonFromDirectory() throws IOException {
        Files.write(tempDir.resolve(LoraAdapterConfig.FILE_NAME), fixtureBytes());

        LoraAdapterConfig config = LoraAdapterConfig.load(tempDir.toFile());

        assertEquals(16, config.rank);
        assertEquals(7, config.targetModules.size());
    }

    @Test
    void rejectsNonPositiveRank() {
        assertThrows(IllegalArgumentException.class,
                () -> new LoraAdapterConfig(0, 16.0, List.of("q_proj")));
        assertThrows(IllegalArgumentException.class,
                () -> new LoraAdapterConfig(-8, 16.0, List.of("q_proj")));
    }

    @Test
    void rejectsEmptyTargetModules() {
        assertThrows(IllegalArgumentException.class,
                () -> new LoraAdapterConfig(16, 16.0, List.of()));
        assertThrows(IllegalArgumentException.class,
                () -> new LoraAdapterConfig(16, 16.0, null));
    }

    private static byte[] fixtureBytes() throws IOException {
        try (InputStream in = LoraAdapterConfigTest.class.getResourceAsStream("/lora/real_shape_adapter_config.json")) {
            return in.readAllBytes();
        }
    }
}
