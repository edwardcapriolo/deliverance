package io.teknek.deliverance.antares;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ToolCallParserTest {
    private final ToolCallParser parser = new ToolCallParser();

    @Test
    void parsesWrappedToolCall() {
        List<ToolCall> calls = parser.parse("thinking<tool_call>{\"name\":\"terminal\",\"arguments\":{\"command\":\"rg -n password .\"}}</tool_call>");

        assertEquals(1, calls.size());
        assertEquals("terminal", calls.get(0).name());
        assertEquals("rg -n password .", calls.get(0).arguments().get("command"));
    }

    @Test
    void parsesRawToolCallWithTrailingText() {
        List<ToolCall> calls = parser.parse("{\"tool\":\"read-file\",\"args\":{\"path\":\"src/App.java\"}} trailing");

        assertEquals(1, calls.size());
        assertEquals("read_file", calls.get(0).name());
        assertEquals("src/App.java", calls.get(0).arguments().get("path"));
    }

    @Test
    void treatsRootRankedFilesAsMalformedSubmitCall() {
        List<ToolCall> calls = parser.parse("<tool_call>{\"ranked_files\":[\"src/main/database.py\"]}</tool_call>");

        assertEquals(1, calls.size());
        assertEquals("submit_vulnerable_files", calls.get(0).name());
        assertEquals(List.of("src/main/database.py"), calls.get(0).arguments().get("ranked_files"));
    }

    @Test
    void treatsNamedRootRankedFilesAsMalformedSubmitCall() {
        List<ToolCall> calls = parser.parse("<tool_call>{\"name\":\"submit_vulnerable_files\",\"ranked_files\":[\"src/main/database.py\"]}</tool_call>");

        assertEquals(1, calls.size());
        assertEquals("submit_vulnerable_files", calls.get(0).name());
        assertEquals(List.of("src/main/database.py"), calls.get(0).arguments().get("ranked_files"));
    }

    @Test
    void stripsToolCallsAndControlTokensFromAssistantText() {
        String cleaned = ToolCallParser.cleanAssistantText("<think>notes</think><tool_call>{}</tool_call>answer<|end_of_text|>");

        assertEquals("notesanswer", cleaned);
    }
}
