package io.teknek.deliverance.nanocode.game;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DeadToRightsGameTest {
    @Test
    void promptKeepsMysteryLightAndNonViolent() {
        String prompt = DeadToRightsGame.systemPrompt();
        String lower = prompt.toLowerCase();

        assertTrue(prompt.contains("Dead to Rights"));
        assertTrue(lower.contains("non-violent"));
        assertTrue(lower.contains("theft"));
        assertTrue(lower.contains("you are the culprit"));
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
                  "crimeDescription": "The donation ledger was stolen.",
                  "meansClue": "a misplaced archive key was found under Mara's desk",
                  "opportunityClue": "the sign-in sheet shows Mara entered after closing",
                  "mistakeClue": "the forged receipt uses the wrong ink color",
                  "hiddenTruth": {
                    "crime": "forgery",
                    "method": "rewrote the receipt and moved the ledger",
                    "mistakes": ["left the key", "used the wrong ink"],
                    "whyCluesMatter": ["receipt shows alteration", "key proves access"]
                  }
                }
                """;

        DeadToRightsGame.CaseFile caseFile = DeadToRightsGame.parseCaseFile(setup);

        assertEquals("The Missing Ledger", caseFile.caseTitle);
        assertTrue(caseFile.publicOpening().contains("CASE TITLE: The Missing Ledger"));
        assertTrue(caseFile.publicOpening().contains("CRIME: The donation ledger was stolen."));
        assertTrue(caseFile.publicOpening().contains("misplaced archive key"));
        assertTrue(caseFile.publicOpening().contains("You may begin questioning me."));
        assertTrue(caseFile.hiddenReveal().contains("CRIME: forgery"));
        assertFalse(caseFile.hiddenReveal().contains("CONFESSION:"));
    }

    @Test
    void caseFileSchemaRequestsPublicAndHiddenTruth() {
        String schema = DeadToRightsGame.caseFileSchema().toString();

        assertTrue(schema.contains("caseTitle"));
        assertTrue(schema.contains("crimeDescription"));
        assertTrue(schema.contains("meansClue"));
        assertTrue(schema.contains("opportunityClue"));
        assertTrue(schema.contains("mistakeClue"));
        assertTrue(schema.contains("hiddenTruth"));
        assertTrue(schema.contains("whyCluesMatter"));
        assertFalse(schema.contains("confession"));
    }

    @Test
    void choicesSchemaRequestsNamedArray() {
        String schema = DeadToRightsGame.choicesSchema("places").toString();

        assertTrue(schema.contains("places"));
        assertTrue(schema.contains("array"));
        assertTrue(schema.contains("string"));
    }
}
