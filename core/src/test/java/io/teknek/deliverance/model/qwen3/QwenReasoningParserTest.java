package io.teknek.deliverance.model.qwen3;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

class QwenReasoningParserTest {
    @Test
    void extractsThinkBlockAndContent() {
        QwenReasoningParser.Parsed parsed = QwenReasoningParser.parse("<think>check carefully</think>Albany", "fallback");

        assertEquals("check carefully", parsed.reasoning());
        assertEquals("Albany", parsed.content());
    }

    @Test
    void fallsBackWhenThinkBlockIsIncomplete() {
        QwenReasoningParser.Parsed parsed = QwenReasoningParser.parse("<think>unfinished", "fallback answer");

        assertNull(parsed.reasoning());
        assertEquals("fallback answer", parsed.content());
    }
}
