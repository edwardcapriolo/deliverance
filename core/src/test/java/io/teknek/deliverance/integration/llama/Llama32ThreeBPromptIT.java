package io.teknek.deliverance.integration.llama;

import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.DoNothingGenerateEvent;
import io.teknek.deliverance.model.LocalGenerationBackend;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class Llama32ThreeBPromptIT {

    @Test
    public void calc() {
        AbstractModel m = Llama32ThreeBSuite.getOrCreate();
        PromptContext ctx = m.promptSupport().get().builder()
                .addSystemMessage("You are an assistant that produces concise, production-grade software.")
                .addSystemMessage("Output java code.")
                .addSystemMessage("Refrain from editorializing your reply.")
                .addSystemMessage("Generate java code into the package 'io.teknek.shape' .")
                .addSystemMessage("Do not import java.awt")
                .addUserMessage("Generate a java interface named Shape with a method named area that returns a double.")
                .addUserMessage("Generate a java class named Circle that extends the Shape interface.")
                .build();

        Response k = m.generate(UUID.randomUUID(), ctx, new GeneratorParameters()
                .withNtokens(2048)
                .withMaxTokens(128)
                .withIncludeStopStrInOutput(false)
                .withStopWords(List.of("<|eot_id|>"))
                .withTemperature(0.2f).withSeed(99998), new DoNothingGenerateEvent());
        assertCrisp(k);
//        assertTrue(k.responseText.contains("public interface Shape"), k.responseText);
        assertTrue(k.responseText.contains("public class Circle extends Shape"), k.responseText);
        assertTrue(k.responseText.contains("public double area()"), k.responseText);
    }

    @Test
    public void nanocodeRootPromptHiNoTools() {
        AbstractModel m = Llama32ThreeBSuite.getOrCreate();
        PromptContext ctx = m.promptSupport().get().builder()
                .addSystemMessage("You are a concise coding assistant. cwd: /tmp. Use tools when needed. Prefer small, direct changes.")
                .addUserMessage("hi")
                .build();

        Response response = m.generate(UUID.randomUUID(), ctx, new GeneratorParameters()
                .withMaxTokens(64)
                .withTemperature(0.0f)
                .withSeed(99999), new DoNothingGenerateEvent());
        assertCrisp(response);
    }

    @Test
    public void factualAnswerIsCorrectAndCrisp() {
        AbstractModel m = Llama32ThreeBSuite.getOrCreate();
        PromptContext ctx = m.promptSupport().get().builder()
                .addUserMessage("What is the capital of France? Answer with one short sentence.")
                .build();

        Response response = m.generate(UUID.randomUUID(), ctx, new GeneratorParameters()
                .withMaxTokens(32)
                .withTemperature(0.0f)
                .withSeed(100001), new DoNothingGenerateEvent());
        assertCrisp(response);
        assertTrue(response.responseText.contains("Paris"), response.responseText);
    }

    @Test
    public void builtinReasoningPromptDoesNotCycle() {
        AbstractModel model = Llama32ThreeBSuite.getOrCreate();
        PromptContext context = model.promptSupport().get().builder()
                .addUserMessage("Read the puzzle carefully and answer with a clear explanation. "
                        + "A company reserves five parking spaces in order for the CEO, president, vice president, "
                        + "secretary, and treasurer. The cars are red, blue, green, yellow, and purple. "
                        + "The first space is red. A blue car is between the red car and the green car. "
                        + "The last space is purple. The secretary drives yellow. Alice parks next to David. "
                        + "Enid drives green. Bert parks between Cheryl and Enid. David parks in the last space. "
                        + "Who is the secretary, and what are the car colors from first to last?")
                .build();

        Response response = model.generateWithBackendRef(UUID.randomUUID(), context, new GeneratorParameters()
                .withMaxTokens(256)
                .withTemperature(0.0f)
                .withSeed(42), (next, nextRaw, nextCleaned, timing) -> {
                    System.err.printf("LLAMA_TOKEN id=%d raw=%s cleaned=%s%n",
                            next, printable(nextRaw), printable(nextCleaned));
                    System.err.flush();
                }, new LocalGenerationBackend(model));

        System.out.println("\nLLAMA_BUILTIN_REASONING_RESPONSE=" + response.responseText.replace("\n", "\\n"));
        String repeated = "It seems like you're trying to solve a puzzle or puzzle.";
        assertTrue(count(response.responseText, repeated) < 2, response.responseText);
    }

    private static void assertNoRepeatedParagraph(String text) {
        String[] paragraphs = text.split("\\R\\s*\\R");
        for (int i = 1; i < paragraphs.length; i++) {
            assertFalse(paragraphs[i].trim().equals(paragraphs[i - 1].trim()), text);
        }
    }

    private static String printable(String value) {
        return value.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r");
    }

    private static int count(String text, String value) {
        int count = 0;
        int offset = 0;
        while ((offset = text.indexOf(value, offset)) >= 0) {
            count++;
            offset += value.length();
        }
        return count;
    }

    private static void assertCrisp(Response response) {
        assertFalse(response.responseText.isBlank(), "Llama response was blank");
        assertFalse(response.responseTextWithSpecialTokens.contains("•\n•\n•"),
                response.responseTextWithSpecialTokens);
        assertFalse(response.responseText.contains("<|"), response.responseText);
        assertTrue(response.generatedTokens.size() >= 1, "No generated tokens");
        if (response.generatedTokens.size() >= 4) {
            int first = response.generatedTokens.getFirst();
            assertFalse(response.generatedTokens.stream().limit(4).allMatch(token -> token == first),
                    "Llama emitted a degenerate repeated-token prefix: " + response.generatedTokens);
        }
    }
}
