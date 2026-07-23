package io.teknek.deliverance.antares;

import org.junit.jupiter.api.Test;

import java.io.IOException;

import static org.junit.jupiter.api.Assertions.assertEquals;

class OpenAiCompletionsClientTest {
    @Test
    void parsesCompletionSseTextChunks() throws IOException {
        String body = "data: {\"choices\":[{\"text\":\"hel\"}]}\n\n"
                + "data: {\"choices\":[{\"text\":\"lo\"}]}\n\n"
                + "data: [DONE]\n\n";

        assertEquals("hello", OpenAiCompletionsClient.parseSse(body));
    }
}
