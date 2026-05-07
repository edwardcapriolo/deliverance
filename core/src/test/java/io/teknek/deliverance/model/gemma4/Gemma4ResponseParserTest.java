package io.teknek.deliverance.model.gemma4;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

public class Gemma4ResponseParserTest {
    @Test
    public void extractsReasoningAndAnswer() {
        Gemma4ResponseParser.Parsed parsed = Gemma4ResponseParser.parse(
                "<|channel>thought\nfirst reason second reason<channel|>Final answer",
                "thought\nfirst reason second reasonFinal answer"
        );
        assertEquals("first reason second reason", parsed.reasoning());
        assertEquals("Final answer", parsed.content());
    }

    @Test
    public void handlesEmptyReasoningBlock() {
        Gemma4ResponseParser.Parsed parsed = Gemma4ResponseParser.parse(
                "<|channel>thought\n<channel|>Final answer",
                "thought\nFinal answer"
        );
        assertNull(parsed.reasoning());
        assertEquals("Final answer", parsed.content());
    }
}
