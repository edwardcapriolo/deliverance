import io.teknek.deliverance.grace.AutoTokenizer;
import io.teknek.deliverance.grace.BytePairEncodingModel;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Method;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

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
}
