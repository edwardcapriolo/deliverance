package io.teknek.deliverance.model;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ReasoningTextSplitterTest {

    @Test
    void consumesSplitThinkTagsFromGeneratedText() {
        ReasoningTextSplitter splitter = new ReasoningTextSplitter();

        assertEquals("", splitter.accept("<think>").content());
        assertEquals("reason", splitter.accept("reason</").reasoning());
        assertEquals("", splitter.accept("think").reasoning());
        ReasoningTextSplitter.Part afterClose = splitter.accept(">answer");

        assertEquals("", afterClose.reasoning());
        assertEquals("answer", afterClose.content());
    }

    @Test
    void startsInsideReasoningWhenPromptAlreadyOpenedThink() {
        ReasoningTextSplitter splitter = new ReasoningTextSplitter(true);

        ReasoningTextSplitter.Part first = splitter.accept("I'll inspect files.");
        ReasoningTextSplitter.Part second = splitter.accept("</");
        ReasoningTextSplitter.Part third = splitter.accept("think>ArchiveController.java");

        assertEquals("", first.content());
        assertEquals("I'll inspect files.", first.reasoning());
        assertEquals("", second.content());
        assertEquals("", second.reasoning());
        assertEquals("ArchiveController.java", third.content());
        assertEquals("", third.reasoning());
    }

    @Test
    void detectsPromptsThatEndInsideReasoning() {
        assertTrue(ReasoningTextSplitter.promptEndsInsideReasoning(
                "<|start_of_role|>assistant<|end_of_role|><think>\n"));
        assertFalse(ReasoningTextSplitter.promptEndsInsideReasoning("<|start_of_role|>assistant<|end_of_role|>"));
    }
}
