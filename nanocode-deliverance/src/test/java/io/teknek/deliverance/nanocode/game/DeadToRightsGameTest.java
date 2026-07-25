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
        assertTrue(prompt.contains("theft"));
        assertTrue(prompt.contains("embezzlement"));
        assertTrue(prompt.contains("forgery"));
        assertTrue(prompt.contains("you are the culprit"));
        assertTrue(prompt.contains("<confession>true</confession>"));
    }

    @Test
    void detectsConfessionMarker() {
        assertTrue(DeadToRightsGame.confessed("Fine, I did it. <confession>true</confession>"));
        assertTrue(DeadToRightsGame.confessed("<CONFESSION>TRUE</CONFESSION>"));
        assertFalse(DeadToRightsGame.confessed("I refuse to confess."));
    }

    @Test
    void extractsTaggedCaseSections() {
        String setup = "before<public>CASE TITLE: Test\n</public><hidden_truth>I took it.</hidden_truth>after";

        assertTrue(DeadToRightsGame.extractTag(setup, "public").contains("CASE TITLE"));
        assertTrue(DeadToRightsGame.extractTag(setup, "hidden_truth").contains("I took it"));
        assertTrue(DeadToRightsGame.extractTag(setup, "missing").isBlank());
    }
}
