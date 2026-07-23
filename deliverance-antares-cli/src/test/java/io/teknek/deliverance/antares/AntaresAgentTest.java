package io.teknek.deliverance.antares;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.List;
import java.util.Queue;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AntaresAgentTest {
    @TempDir
    Path tempDir;

    @Test
    void fallsBackToObservedCandidateFilesWhenModelDoesNotSubmit() throws Exception {
        Path source = tempDir.resolve("src/main");
        Files.createDirectories(source);
        Files.writeString(source.resolve("database.py"), "db.query('select ' + user)\n");
        FakeClient client = new FakeClient(List.of(
                "<tool_call>{\"name\":\"terminal\",\"arguments\":{\"command\":\"cat src/main/database.py\",\"max_chars\":0}}</tool_call>",
                "I found the SQL injection in src/main/database.py and will submit it now.",
                "I will submit the ranked list now.",
                "Still preparing to submit."));
        AntaresAgent agent = new AntaresAgent(client, new AntaresToolExecutor(tempDir, 1), 10, 1);

        AgentResult result = agent.run("Search for CWE-89 SQL injection");

        assertTrue(result.vulnerabilityFound());
        assertEquals(List.of("src/main/database.py"), result.rankedFiles());
        assertTrue(result.summary().contains("did not emit submit_vulnerable_files")
                || result.summary().contains("tool budget"));
    }

    private static final class FakeClient implements CompletionClient {
        private final Queue<String> turns;

        private FakeClient(List<String> turns) {
            this.turns = new ArrayDeque<>(turns);
        }

        @Override
        public String complete(List<Message> messages, Consumer<String> onChunk) {
            String next = turns.remove();
            onChunk.accept(next);
            return next;
        }
    }
}
