package io.teknek.deliverance.nanocode.game;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DeadToRightsGameTest {
    @Test
    void promptKeepsMysteryLightAndNonViolent() {
        String prompt = DeadToRightsGame.systemPrompt();

        assertTrue(prompt.contains("Dead to Rights"));
        assertTrue(prompt.contains("non-violent"));
        assertTrue(prompt.contains("must not involve anyone being hurt"));
        assertTrue(prompt.contains("theft"));
        assertTrue(prompt.contains("embezzlement"));
        assertTrue(prompt.contains("you are the culprit"));
        assertTrue(prompt.contains("<confession>true</confession>"));
    }

    @Test
    void detectsConfessionMarker() {
        assertTrue(DeadToRightsGame.confessed("Fine, I did it. <confession>true</confession>"));
        assertTrue(DeadToRightsGame.confessed("<CONFESSION>TRUE</CONFESSION>"));
        assertFalse(DeadToRightsGame.confessed("I refuse to confess."));
    }
}
