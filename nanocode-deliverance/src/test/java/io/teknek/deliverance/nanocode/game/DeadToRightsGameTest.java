package io.teknek.deliverance.nanocode.game;

import org.junit.jupiter.api.Test;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DeadToRightsGameTest {
    private static final ObjectMapper JSON = new ObjectMapper();

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
    void cleansVisibleThinkingMarkers() {
        assertEquals("I was at lunch.", DeadToRightsGame.cleanVisibleText("</think>\n\nI was at lunch."));
        assertEquals("Final answer.", DeadToRightsGame.cleanVisibleText("<think>hidden reasoning</think>Final answer."));
    }

    @Test
    void readsStreamingFinishReason() throws Exception {
        assertEquals("length", DeadToRightsGame.finishReason(JSON.readTree("""
                {"choices":[{"index":0,"delta":{},"finish_reason":"length"}]}
                """)));
        assertEquals("", DeadToRightsGame.finishReason(JSON.readTree("""
                {"choices":[{"index":0,"delta":{"content":"hello"}}]}
                """)));
    }

    @Test
    void parsesStructuredCaseFile() throws Exception {
        String setup = """
                {
                  "caseTitle": "The Missing Ledger",
                  "suspect": "Mara Vale, bookkeeper",
                  "setting": "community theater",
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
        assertTrue(caseFile.publicOpening().contains("misplaced archive key"));
        assertTrue(caseFile.publicOpening().contains("You may begin questioning me."));
        assertTrue(caseFile.hiddenReveal().contains("CRIME: forgery"));
        assertFalse(caseFile.hiddenReveal().contains("CONFESSION:"));
    }

    @Test
    void parseFailureIncludesModelOutput() {
        IOException thrown = assertThrows(IOException.class,
                () -> DeadToRightsGame.parseCaseFile("{\"caseTitle\":\"The Almost Case\", \"suspect\":"));

        assertTrue(thrown.getMessage().contains("Could not parse Dead to Rights case JSON"));
        assertTrue(thrown.getMessage().contains("The Almost Case"));
        assertFalse(thrown.getCause().getMessage().contains("Source: REDACTED"));
        assertTrue(thrown.getCause().getMessage().contains("The Almost Case"));
    }

    @Test
    void caseFileSchemaRequestsPublicAndHiddenTruth() {
        String schema = DeadToRightsGame.caseFileSchema().toString();

        assertTrue(schema.contains("caseTitle"));
        assertFalse(schema.contains("crimeDescription"));
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

    @Test
    void cleansGeneratedSetupChoices() {
        assertEquals("ledger", DeadToRightsGame.cleanChoice(",ledger"));
        assertEquals("donation envelope", DeadToRightsGame.cleanChoice(",donation_envelope"));
        assertEquals("camera", DeadToRightsGame.cleanChoice("  .camera"));
    }

    @Test
    void suspectNameRegexRejectsMarkdownDecoratedNames() {
        assertTrue("Luna".matches(DeadToRightsGame.SUSPECT_NAME_REGEX));
        assertTrue("Mara Vale".matches(DeadToRightsGame.SUSPECT_NAME_REGEX));
        assertFalse("**Luna**".matches(DeadToRightsGame.SUSPECT_NAME_REGEX));
    }
}
