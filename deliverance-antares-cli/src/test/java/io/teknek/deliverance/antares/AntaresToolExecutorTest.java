package io.teknek.deliverance.antares;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AntaresToolExecutorTest {
    @TempDir
    Path tempDir;

    @Test
    void readFileRequiresRepoContainment() throws Exception {
        Files.writeString(tempDir.resolve("safe.txt"), "one\ntwo\nthree\n");
        AntaresToolExecutor executor = new AntaresToolExecutor(tempDir, 5);

        ToolResult result = executor.execute(new ToolCall("read_file", Map.of(
                "path", "safe.txt",
                "start_line", 2,
                "end_line", 3)));

        assertTrue(result.response().contains("2: two"));
        assertTrue(result.response().contains("3: three"));
        assertTrue(executor.execute(new ToolCall("read_file", Map.of("path", "../outside.txt")))
                .response().contains("path escapes repository root"));
    }

    @Test
    void terminalMaxCharsZeroUsesDefaultOutputLimit() throws Exception {
        Files.writeString(tempDir.resolve("safe.txt"), "secret\n");
        AntaresToolExecutor executor = new AntaresToolExecutor(tempDir, 5);

        ToolResult result = executor.execute(new ToolCall("terminal", Map.of(
                "command", "cat safe.txt",
                "max_chars", 0)));

        assertTrue(result.response().contains("secret"));
    }

    @Test
    void terminalCommandCanBeDeclinedByApprover() throws Exception {
        AntaresToolExecutor executor = new AntaresToolExecutor(tempDir, 5, (repoRoot, command) -> false);

        ToolResult result = executor.execute(new ToolCall("terminal", Map.of("command", "echo should-not-run")));

        assertTrue(result.response().contains("declined by user"));
    }

    @Test
    void approvedTerminalCommandRunsWithoutAllowlistBlocking() throws Exception {
        AntaresToolExecutor executor = new AntaresToolExecutor(tempDir, 5, (repoRoot, command) -> true);

        ToolResult result = executor.execute(new ToolCall("terminal", Map.of("command", "echo approved")));

        assertTrue(result.response().contains("approved"));
    }

    @Test
    void emptyRipgrepResultExplainsNoMatches() throws Exception {
        Files.writeString(tempDir.resolve("safe.txt"), "secret\n");
        AntaresToolExecutor executor = new AntaresToolExecutor(tempDir, 5, (repoRoot, command) -> true);

        ToolResult result = executor.execute(new ToolCall("terminal", Map.of("command", "rg -n missing-token .")));

        assertTrue(result.response().contains("ripgrep found no matches"));
    }

    @Test
    void submitRejectsGlobsAndPlaceholdersAndAcceptsExistingFiles() throws Exception {
        Files.createDirectories(tempDir.resolve("src/main/java/demo"));
        Files.writeString(tempDir.resolve("src/main/java/demo/ArchiveController.java"), "class ArchiveController {}\n");
        AntaresToolExecutor executor = new AntaresToolExecutor(tempDir, 5);

        ToolResult glob = executor.execute(new ToolCall("submit_vulnerable_files", Map.of(
                "ranked_files", java.util.List.of("*dao*", "path/to/file"))));
        ToolResult valid = executor.execute(new ToolCall("submit_vulnerable_files", Map.of(
                "ranked_files", java.util.List.of("src/main/java/demo/ArchiveController.java"))));

        assertTrue(glob.response().contains("must be existing repository-relative file paths"));
        assertTrue(valid.submitted());
        assertEquals(java.util.List.of("src/main/java/demo/ArchiveController.java"), valid.result().rankedFiles());
    }
}
