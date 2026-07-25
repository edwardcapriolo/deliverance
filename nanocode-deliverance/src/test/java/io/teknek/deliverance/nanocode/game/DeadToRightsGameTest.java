package io.teknek.deliverance.nanocode.game;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DeadToRightsGameTest {
    @Test
    void promptKeepsMysteryLightAndNonViolent() {
        String prompt = DeadToRightsGame.systemPrompt();

        assertTrue(prompt.contains("Dead to Rights"));
        assertTrue(prompt.contains("non-violent"));
        assertTrue(prompt.contains("theft"));
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
    void parsesStructuredCaseFile() throws Exception {
        String setup = """
                {
                  "caseTitle": "The Missing Ledger",
                  "suspect": "Mara Vale, bookkeeper",
                  "setting": "community theater",
                  "clues": ["a rewritten receipt", "a misplaced key", "muddy footprints"],
                  "hiddenTruth": {
                    "crime": "forgery",
                    "method": "rewrote the receipt and moved the ledger",
                    "mistakes": ["left the key", "used the wrong ink"],
                    "whyCluesMatter": ["receipt shows alteration", "key proves access"],
                    "confession": "I confess. I forged it because I needed time."
                  }
                }
                """;

        DeadToRightsGame.CaseFile caseFile = DeadToRightsGame.parseCaseFile(setup);

        assertEquals("The Missing Ledger", caseFile.caseTitle);
        assertTrue(caseFile.publicOpening().contains("CASE TITLE: The Missing Ledger"));
        assertTrue(caseFile.publicOpening().contains("You may begin questioning me."));
        assertTrue(caseFile.hiddenReveal().contains("CRIME: forgery"));
        assertTrue(caseFile.hiddenReveal().contains("I confess"));
    }

    @Test
    void caseFileSchemaRequestsPublicAndHiddenTruth() {
        String schema = DeadToRightsGame.caseFileSchema().toString();

        assertTrue(schema.contains("caseTitle"));
        assertTrue(schema.contains("clues"));
        assertTrue(schema.contains("hiddenTruth"));
        assertTrue(schema.contains("confession"));
    }
}
