package io.teknek.deliverance.antares;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AntaresPromptTest {
    @Test
    void rendersGranitePromptForCompletions() {
        String prompt = AntaresPrompt.render(List.of(
                new Message("system", "sys"),
                new Message("user", "hi <|end_of_text|>")));

        assertTrue(prompt.contains("<|start_of_role|>system<|end_of_role|>sys<|end_of_text|>"));
        assertTrue(prompt.endsWith("<|start_of_role|>assistant<|end_of_role|><think>\n"));
        assertFalse(prompt.contains("hi <|end_of_text|><|end_of_text|>"));
        assertTrue(prompt.contains("[escaped Granite control token: end_of_text]"));
    }

    @Test
    void rendersToolResponsesAsGraniteToolResponseTurns() {
        String prompt = AntaresPrompt.render(List.of(
                new Message("system", "sys"),
                AntaresPrompt.toolResponse("result")));

        assertTrue(prompt.contains("<|start_of_role|>user<|end_of_role|>\n<tool_response>\nresult\n</tool_response><|end_of_text|>"));
    }

    @Test
    void cwePromptIncludesCwe78Context() {
        String prompt = CwePrompts.analysisPrompt("CWE-78", "extra");

        assertTrue(prompt.contains("CWE-78: Improper Neutralization"));
        assertTrue(prompt.contains("Likelihood of Exploit: High"));
        assertTrue(prompt.contains("Additional instructions:\nextra"));
    }
}
