package io.teknek.deliverance.integration;

import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.*;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertFalse;

@Tag("large-model")
class Qwen3BenchmarkCasesIT {

    @Test
    void qwen34BJq4FirstTwoBenchmarkCases() {
        boolean previousProfiling = InferenceProfiler.isEnabled();
        InferenceProfiler.setEnabled(true);
        ModelFetcher fetch = new ModelFetcher("edwardcapriolo", "Qwen3-4B-JQ4").withDownload(false);
        Assumptions.assumeTrue(fetch.pathForModel().toFile().isDirectory(),
                "Qwen3-4B-JQ4 cache is not present: " + fetch.pathForModel());

        try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetch)
                .withConfig(AutoModelConfig.fromJson(resolveRepoPath("benchmarks/configs/qwen3-4b-jq4.json")))
                .withDownload(false).withDecodeAttentionMode(DecodeAttentionMode.FLASH_DECODE)
                .buildLocalTransformerModel()) {
            runCase(model, "builtin-reasoning-1", List.of(
                    "Read the puzzle carefully and answer with a clear explanation. A company reserves five parking spaces in order for the CEO, president, vice president, secretary, and treasurer. The cars are red, blue, green, yellow, and purple. The first space is red. A blue car is between the red car and the green car. The last space is purple. The secretary drives yellow. Alice parks next to David. Enid drives green. Bert parks between Cheryl and Enid. David parks in the last space. Who is the secretary, and what are the car colors from first to last?",
                    "Now explain which clues were necessary and which were redundant."));
            runCase(model, "builtin-math-1", List.of(
                    "Solve step by step. A bus starts with an unknown number of passengers. At the first stop, half get off and 4 get on. At the second stop, 6 get off and 8 get on. There are 25 passengers heading to the third stop. How many passengers started at the terminal? Then compute total fare collected if every person who ever boarded paid $2.",
                    "Generalize the algebra for starting passengers S, first-stop additions A, second-stop exits B, second-stop additions C, and final passengers F."));
        } finally {
            InferenceProfiler.setEnabled(previousProfiling);
        }
    }

    private static void runCase(AbstractModel model, String caseId, List<String> turns) {
        List<ChatMessage> messages = new ArrayList<>();
        for (int turn = 0; turn < turns.size(); turn++) {
            messages.add(new ChatMessage("user", turns.get(turn)));
            PromptContext prompt = promptContext(model.promptSupport(), messages);
            InferenceProfiler.reset();
            Response response = model.generate(UUID.randomUUID(), prompt,
                    new GeneratorParameters().withTemperature(0.0f).withMaxTokens(256).withSeed(42),
                    new DoNothingGenerateEvent());
            messages.add(new ChatMessage("assistant", response.responseText));
            double generationMs = Math.max(0.0, response.totalTimeMs - response.timeToFirstTokenMs);
            long decodeTokens = Math.max(0, response.generatedTokens.size() - 1L);
            double tokensPerSecond = generationMs == 0.0 ? 0.0 : decodeTokens / (generationMs / 1000.0);
            System.out.printf(java.util.Locale.ROOT,
                    "[qwen-benchmark-it] case=%s turn=%d prompt_tokens=%d generated=%d total_ms=%.1f tok_s=%.2f finish=%s%n",
                    caseId, turn + 1, response.promptTokens, response.generatedTokens.size(), response.totalTimeMs,
                    tokensPerSecond, response.finishReason);
            InferenceProfiler.printSummary("case=" + caseId + " turn=" + (turn + 1), 30);
            model.getMetricRegistry().getCounters().entrySet().stream()
                    .filter(entry -> InferenceProfiler.shouldPrintCounter(entry.getKey())
                            || entry.getKey().startsWith("tensorplan.")
                            || entry.getKey().startsWith("gpu."))
                    .forEach(entry -> {
                        long count = InferenceProfiler.shouldPrintCounter(entry.getKey())
                                ? InferenceProfiler.counterValue(entry.getKey())
                                : entry.getValue().getCount();
                        System.out.println("[profile-counter] " + entry.getKey() + " count=" + count);
                    });
            assertFalse(response.responseTextWithSpecialTokens.isBlank());
        }
    }

    private static PromptContext promptContext(Optional<PromptSupport> promptSupport, List<ChatMessage> messages) {
        if (promptSupport.isPresent()) {
            PromptSupport.Builder builder = promptSupport.get().builder();
            for (ChatMessage message : messages) {
                switch (message.role()) {
                    case "user" -> builder.addUserMessage(message.content());
                    case "assistant" -> builder.addAssistantMessage(message.content());
                    case "system" -> builder.addSystemMessage(message.content());
                    default -> throw new IllegalArgumentException("Unsupported role " + message.role());
                }
            }
            return builder.build();
        }
        StringBuilder raw = new StringBuilder();
        for (ChatMessage message : messages) {
            raw.append(message.role()).append(": ").append(message.content()).append('\n');
        }
        raw.append("assistant: ");
        return PromptContext.of(raw.toString());
    }

    private record ChatMessage(String role, String content) {
    }

    private static Path resolveRepoPath(String relativePath) {
        Path current = Path.of("").toAbsolutePath();
        while (current != null) {
            Path candidate = current.resolve(relativePath);
            if (Files.exists(candidate)) {
                return candidate;
            }
            current = current.getParent();
        }
        throw new IllegalStateException("Unable to locate " + relativePath + " from " + Path.of("").toAbsolutePath());
    }
}
