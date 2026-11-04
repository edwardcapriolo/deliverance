package io.teknek.reckoning.model;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DeadReckoningTraceCheckTest {
    @TempDir
    Path tempDir;

    @Test
    void exactSemanticRecordsMatchEvenWhenJsonlOrderDiffers() throws Exception {
        Path expected = write("expected.jsonl", """
                {"kind":"CHECKPOINT","phase":"PREFILL","position":-1,"layer":0,"stage":"input","name":"layer.0.input","mean":1.0,"hash":"a"}
                {"kind":"RUN_START","model":"qwen"}
                """);
        Path actual = write("actual.jsonl", """
                {"kind":"RUN_START","model":"qwen"}
                {"kind":"CHECKPOINT","phase":"PREFILL","position":-1,"layer":0,"stage":"input","name":"layer.0.input","mean":1.0,"hash":"a"}
                """);

        assertTrue(DeadReckoningTraceCheck.same(expected, actual));
    }

    @Test
    void changedStatisticReturnsFalseAndExplainsDifference() throws Exception {
        Path expected = write("expected.jsonl", """
                {"kind":"CHECKPOINT","phase":"DECODE","position":18,"layer":-1,"stage":"input","name":"layer.-1.input","mean":1.0,"hash":"a"}
                """);
        Path actual = write("actual.jsonl", """
                {"kind":"CHECKPOINT","phase":"DECODE","position":18,"layer":-1,"stage":"input","name":"layer.-1.input","mean":1.25,"hash":"b"}
                """);

        assertFalse(DeadReckoningTraceCheck.same(expected, actual));
        String explanation = DeadReckoningTraceCheck.explain(expected, actual);
        assertTrue(explanation.contains("layer.-1.input"));
        assertTrue(explanation.contains("mean"));
        assertTrue(explanation.contains("1.0"));
        assertTrue(explanation.contains("1.25"));
    }

    private Path write(String name, String contents) throws Exception {
        Path path = tempDir.resolve(name);
        Files.writeString(path, contents);
        return path;
    }
}
