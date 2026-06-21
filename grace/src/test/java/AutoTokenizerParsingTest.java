import io.teknek.deliverance.grace.AutoTokenizer;
import io.teknek.deliverance.grace.BytePairEncodingModel;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Method;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

public class AutoTokenizerParsingTest {
    @Test
    void parsesMergePairsFromTwoElementArrays() throws Exception {
        String mergesJson = """
                [["a","b"],["ab","c"]]
                """;
        com.fasterxml.jackson.databind.ObjectMapper om = new com.fasterxml.jackson.databind.ObjectMapper();
        var node = om.readTree(mergesJson);

        Method readMerges = AutoTokenizer.class.getDeclaredMethod("readMerges", com.fasterxml.jackson.databind.JsonNode.class);
        readMerges.setAccessible(true);
        @SuppressWarnings("unchecked")
        List<String> merges = (List<String>) readMerges.invoke(null, node);

        assertEquals(List.of("a b", "ab c"), merges);
        assertEquals(BytePairEncodingModel.fromMerges(merges).get("a b"), 0);
        assertEquals(BytePairEncodingModel.fromMerges(merges).get("ab c"), 1);
    }

    @Test
    void parsesBpeModelWithoutTypeWhenVocabAndMergesArePresent() throws Exception {
        String tokenizerJson = """
                {
                  "model": {
                    "vocab": {"a": 0, "b": 1, "ab": 2},
                    "merges": ["a b"]
                  }
                }
                """;
        com.fasterxml.jackson.databind.ObjectMapper om = new com.fasterxml.jackson.databind.ObjectMapper();
        var tokenizerDocument = om.readTree(tokenizerJson);

        Method parseBytePairEncodingModel = AutoTokenizer.class.getDeclaredMethod(
                "parseBytePairEncodingModel", java.io.File.class, com.fasterxml.jackson.databind.JsonNode.class, Map.class);
        parseBytePairEncodingModel.setAccessible(true);
        BytePairEncodingModel model = (BytePairEncodingModel) parseBytePairEncodingModel.invoke(
                null, new java.io.File("."), tokenizerDocument, Map.of("a", 0, "b", 1, "ab", 2));

        assertNotNull(model);
        assertEquals(0, model.mergeRanks().get("a b"));
    }
}
