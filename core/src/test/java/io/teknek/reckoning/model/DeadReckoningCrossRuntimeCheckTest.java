package io.teknek.reckoning.model;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;

class DeadReckoningCrossRuntimeCheckTest {
    @TempDir
    Path tempDir;

    @Test
    void mapsKnownStagesAndAcceptsSmallDrift() throws Exception {
        Path hf = write("hf.jsonl", checkpoint("model.embed_tokens", "[1,2]", 1.0));
        Path java = write("java.jsonl", checkpoint("layer.-1.input", "[2]", 1.0001));

        DeadReckoningCrossRuntimeCheck.Result result = DeadReckoningCrossRuntimeCheck.compare(
                hf, java, DeadReckoningCrossRuntimeCheck.Policy.causalDefaults());

        assertEquals(DeadReckoningCrossRuntimeCheck.Status.GREEN, result.status());
        assertEquals(1, result.matchedStages());
    }

    @Test
    void flagsLargeDriftRed() throws Exception {
        Path hf = write("hf.jsonl", checkpoint("model.embed_tokens", "[1,2]", 1.0));
        Path java = write("java.jsonl", checkpoint("layer.-1.input", "[2]", 2.0));

        DeadReckoningCrossRuntimeCheck.Result result = DeadReckoningCrossRuntimeCheck.compare(
                hf, java, DeadReckoningCrossRuntimeCheck.Policy.causalDefaults());

        assertEquals(DeadReckoningCrossRuntimeCheck.Status.RED, result.status());
    }

    private Path write(String name, String contents) throws Exception {
        Path path = tempDir.resolve(name);
        Files.writeString(path, contents);
        return path;
    }

    private static String checkpoint(String name, String shape, double mean) {
        return "{\"kind\":\"CHECKPOINT\",\"phase\":\"PREFILL\",\"position\":-1,"
                + "\"name\":\"" + name + "\",\"shape\":" + shape
                + ",\"count\":2,\"min\":1,\"max\":1,\"mean\":" + mean
                + ",\"sum\":2,\"mean_abs\":1,\"max_abs\":1,\"l2_norm\":1,"
                + "\"rms\":1,\"stddev\":0,\"zero_count\":0,\"nan_count\":0,"
                + "\"positive_infinity_count\":0,\"negative_infinity_count\":0}\n";
    }
}
