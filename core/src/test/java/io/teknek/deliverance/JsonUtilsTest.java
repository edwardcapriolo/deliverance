package io.teknek.deliverance;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class JsonUtilsTest {

    @Test
    void mapperOmitsNullValues() throws Exception {
        String json = JsonUtils.om.writeValueAsString(Map.of("property", new NullableField("string", null)));

        assertTrue(json.contains("type"));
        assertFalse(json.contains("description"));
    }

    private record NullableField(String type, String description) {
    }
}
