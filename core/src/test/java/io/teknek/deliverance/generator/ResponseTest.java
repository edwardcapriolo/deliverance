package io.teknek.deliverance.generator;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ResponseTest {

    @Test
    void copyWithTimingPopulatesLegacyAndDetailedTimingFields() {
        Response response = new Response("a", "a", FinishReason.STOP_TOKEN, 3, List.of(1, 2), 0, 0, List.of());

        Response timed = response.copyWithTiming(12.4, 6.2, 18.8);

        assertEquals(12.4, timed.timeToFirstTokenMs);
        assertEquals(6.2, timed.avgTimePerTokenMs);
        assertEquals(18.8, timed.totalTimeMs);
        assertEquals(12, timed.promptTimeMs);
        assertEquals(19, timed.generateTimeMs);
    }
}
