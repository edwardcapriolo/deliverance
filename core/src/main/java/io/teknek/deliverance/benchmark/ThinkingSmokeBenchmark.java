package io.teknek.deliverance.benchmark;

import com.fasterxml.jackson.databind.node.ObjectNode;
import io.teknek.deliverance.JsonUtils;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelConfig;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.model.GenerateEvent;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.PromptContext;

import java.io.BufferedWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.UUID;
import java.util.function.Predicate;
import java.util.ArrayList;

/**
 * Small qualitative smoke benchmark for thinking-capable chat models.
 *
 * <p>This is not a throughput benchmark. It checks whether a model can still complete a small pack of understandable,
 * bounded prompts and records only pass/fail, finish reason, generated tokens, time, and response.</p>
 */
public final class ThinkingSmokeBenchmark {
    private static final String RESET = "\033[0m";
    private static final String BOLD = "\033[1m";
    private static final String BLUE = "\033[34m";
    private static final String CYAN = "\033[36m";
    private static final String DIM = "\033[2m";

    private ThinkingSmokeBenchmark() {
    }

    public static void main(String[] args) throws Exception {
        Options options = Options.parse(args);
        if (options.output.getParent() != null) {
            Files.createDirectories(options.output.getParent());
        }
        try (BufferedWriter writer = Files.newBufferedWriter(options.output, StandardCharsets.UTF_8)) {
            ModelFetcher fetcher = new ModelFetcher(options.owner, options.model);
            AutoModelForCausaLm.Builder builder = AutoModelForCausaLm.newBuilder(fetcher)
                    .withWrappedForkJoinPool(new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores()));
            if (options.modelConfig != null) {
                builder.withConfig(AutoModelConfig.fromJson(options.modelConfig));
            }
            try (AbstractModel model = builder.buildLocalTransformerModel()) {
                int passed = 0;
                int maxTokenWarnings = 0;
                List<SmokeCase> cases = smokeCases();
                for (SmokeCase smokeCase : cases) {
                    List<ChatMessage> messages = new ArrayList<>();
                    long totalTimeMs = 0;
                    int totalGeneratedTokens = 0;
                    Response response = null;
                    System.out.println();
                    System.out.println(DIM + "================================================================================" + RESET);
                    System.out.println(BOLD + BLUE + "case: " + smokeCase.id + RESET);
                    int turns = smokeCase.chineseRoundTrip ? 3 : smokeCase.prompts.size();
                    String chineseQuestion = null;
                    String chineseAnswer = null;
                    for (int turn = 0; turn < turns; turn++) {
                        String userPrompt = promptForTurn(smokeCase, turn, chineseQuestion, chineseAnswer);
                        List<ChatMessage> turnMessages;
                        if (smokeCase.chineseRoundTrip) {
                            turnMessages = List.of(new ChatMessage("user", userPrompt));
                        } else {
                            messages.add(new ChatMessage("user", userPrompt));
                            turnMessages = messages;
                        }
                        System.out.println(BOLD + CYAN + "question " + (turn + 1) + "/" + turns
                                + ":" + RESET);
                        System.out.println(userPrompt);
                        System.out.println(DIM + "--------------------------------------------------------------------------------" + RESET);
                        System.out.println(BOLD + CYAN + "answer " + (turn + 1) + "/" + turns
                                + ":" + RESET);
                        PromptContext prompt = promptContext(model, turnMessages, options.enableThinking);
                        long start = System.nanoTime();
                        response = model.generate(UUID.randomUUID(), prompt,
                                new GeneratorParameters()
                                        .withTemperature(options.temperature)
                                        .withMaxTokens(options.maxTokens),
                                new StreamingPrinter());
                        System.out.println();
                        long timeMs = (System.nanoTime() - start) / 1_000_000L;
                        totalTimeMs += timeMs;
                        totalGeneratedTokens += response.generatedTokens == null ? 0 : response.generatedTokens.size();
                        if (!smokeCase.chineseRoundTrip) {
                            messages.add(new ChatMessage("assistant", response.responseText));
                        }
                        if (smokeCase.chineseRoundTrip && turn == 0) {
                            chineseQuestion = cleanModelAnswer(response.responseTextWithSpecialTokens);
                        } else if (smokeCase.chineseRoundTrip && turn == 1) {
                            chineseAnswer = cleanModelAnswer(response.responseTextWithSpecialTokens);
                        }
                        System.out.println(DIM + "turn_result: finish="
                                + (response.finishReason == null ? "" : response.finishReason.name())
                                + " generated=" + (response.generatedTokens == null ? 0 : response.generatedTokens.size())
                                + " time_ms=" + timeMs + RESET);
                        System.out.println(DIM + "--------------------------------------------------------------------------------" + RESET);
                    }
                    String normalized = normalize(response == null ? "" : response.responseTextWithSpecialTokens);
                    boolean pass = smokeCase.check.test(normalized);
                    if (pass) {
                        passed++;
                    }
                    if (response != null && response.finishReason != null && "MAX_TOKENS".equals(response.finishReason.name())) {
                        maxTokenWarnings++;
                    }
                    ObjectNode row = JsonUtils.om.createObjectNode();
                    row.put("timestamp", Instant.now().toString());
                    row.put("case_id", smokeCase.id);
                    row.put("pass", pass);
                    row.put("finish_reason", response == null || response.finishReason == null ? "" : response.finishReason.name());
                    row.put("generated_tokens", totalGeneratedTokens);
                    row.put("time_ms", totalTimeMs);
                    row.put("time_to_first_token_ms", response == null ? 0.0d : response.timeToFirstTokenMs);
                    row.put("response", response == null ? "" : response.responseTextWithSpecialTokens);
                    writer.write(JsonUtils.om.writeValueAsString(row));
                    writer.newLine();
                    writer.flush();
                    System.out.println(DIM + "--------------------------------------------------------------------------------" + RESET);
                    System.out.printf(Locale.ROOT,
                            BOLD + BLUE + "result:" + RESET + " pass=%s finish=%s generated=%d time_ms=%d ttft_ms=%.3f%n",
                            pass,
                            response == null || response.finishReason == null ? "" : response.finishReason.name(),
                            totalGeneratedTokens,
                            totalTimeMs,
                            response == null ? 0.0d : response.timeToFirstTokenMs);
                }
                System.out.printf(Locale.ROOT,
                        BOLD + BLUE + "summary" + RESET + " passed=%d/%d max_token_warnings=%d output=%s%n",
                        passed,
                        cases.size(),
                        maxTokenWarnings,
                        options.output.toAbsolutePath());
            }
        }
    }

    private static List<SmokeCase> smokeCases() {
        return List.of(
                new SmokeCase("square-area",
                        "I have a square with a side length of 4. What is its area? Think if needed, then answer with `final: <answer>`.",
                        containsAll("final:", "16")),
                new SmokeCase("bat-ball",
                        "A bat and a ball cost $1.10 together. The bat costs $1.00 more than the ball. How much does the ball cost? Think if needed, then answer with `final: <answer>`.",
                        containsAny("final: 5", "final: $0.05", "final: 0.05", "final: five cents")),
                new SmokeCase("calendar",
                        "Today is Wednesday. I have a meeting 10 days after the day before tomorrow. What day of the week is the meeting? Think if needed, then answer with `final: <day>`.",
                        containsAll("final:", "saturday")),
                new SmokeCase("chinese-roundtrip",
                        List.of(
                                "Translate this question into Chinese only: I have a square with a side length of 4. What is its area?"
                        ),
                        containsAll("final:", "16"),
                        true),
                new SmokeCase("java-syntax",
                        "The following Java method does not compile. What is the syntax error? Think if needed, then answer with `final: <one sentence>`.\n\npublic int add(int a, int b) {\n    return a + b\n}",
                        containsAll("final:", "semicolon")),
                new SmokeCase("odd-one-out",
                        "Choose the word that does not belong: apple, banana, carrot, grape. Think if needed, then answer with `final: <word>`.",
                        containsAll("final:", "carrot")),
                new SmokeCase("syllogism",
                        "If all daxes are wugs, and no wugs are nims, can a dax be a nim? Think if needed, then answer with `final: yes` or `final: no`.",
                        containsAll("final:", "no")),
                new SmokeCase("missing-dollar",
                        "Three people pay $30 for a room. The clerk realizes it should cost $25 and sends back $5. The bellhop keeps $2 and gives each person $1 back. Each person paid $9, totaling $27, and the bellhop kept $2, totaling $29. Where is the missing dollar? Think if needed, then answer with `final: <one sentence>`.",
                        containsAny("no missing", "not missing", "there is no missing", "27 includes")),
                new SmokeCase("tool-choice",
                        "Available tools: read reads one file, glob lists files by pattern, grep searches inside text files. User asks: find lines containing `needle` in README.md. Think if needed, then answer with `final: <tool name>`.",
                        containsAll("final:", "grep"))
        );
    }

    private static String promptForTurn(SmokeCase smokeCase, int turn, String chineseQuestion, String chineseAnswer) {
        if (!smokeCase.chineseRoundTrip) {
            return smokeCase.prompts.get(turn);
        }
        return switch (turn) {
            case 0 -> smokeCase.prompts.get(0);
            case 1 -> cleanOrFallback(chineseQuestion,
                    "我有一个边长为4的正方形。它的面积是多少？") + "\n请只用中文思考并回答。";
            case 2 -> "Translate this Chinese answer back to English. End with `final: <answer>`\n\n"
                    + cleanOrFallback(chineseAnswer, "面积是16平方单位。");
            default -> throw new IllegalArgumentException("unexpected turn " + turn);
        };
    }

    private static String cleanOrFallback(String value, String fallback) {
        return value == null || value.isBlank() ? fallback : value;
    }

    private static String cleanModelAnswer(String response) {
        if (response == null) {
            return "";
        }
        String cleaned = response.replaceAll("(?s)<think>.*?</think>", "")
                .replace("<|im_end|>", "")
                .trim();
        int finalIndex = cleaned.toLowerCase(Locale.ROOT).lastIndexOf("final:");
        if (finalIndex >= 0) {
            cleaned = cleaned.substring(finalIndex + "final:".length()).trim();
        }
        return cleaned;
    }

    private static PromptContext promptContext(AbstractModel model, List<ChatMessage> messages, boolean enableThinking) {
        var builder = model.promptSupport().orElseThrow().builder()
                .addTemplateArgs(Map.of("enable_thinking", enableThinking));
        for (ChatMessage message : messages) {
            if ("user".equals(message.role)) {
                builder.addUserMessage(message.content);
            } else if ("assistant".equals(message.role)) {
                builder.addAssistantMessage(message.content);
            }
        }
        return builder.build();
    }

    private static Predicate<String> containsAll(String... needles) {
        return value -> {
            for (String needle : needles) {
                if (!value.contains(needle.toLowerCase(Locale.ROOT))) {
                    return false;
                }
            }
            return true;
        };
    }

    private static Predicate<String> containsAny(String... needles) {
        return value -> {
            for (String needle : needles) {
                if (value.contains(needle.toLowerCase(Locale.ROOT))) {
                    return true;
                }
            }
            return false;
        };
    }

    private static String normalize(String value) {
        return value == null ? "" : value.toLowerCase(Locale.ROOT).replace("`", "").trim();
    }

    private static final class StreamingPrinter implements GenerateEvent {
        @Override
        public void emit(int next, String nextRaw, String nextCleaned, float timing) {
            if (nextCleaned != null && !nextCleaned.isEmpty()) {
                System.out.print(nextCleaned);
                System.out.flush();
            }
        }
    }

    private record ChatMessage(String role, String content) {
    }

    private record SmokeCase(String id, List<String> prompts, Predicate<String> check, boolean chineseRoundTrip) {
        private SmokeCase(String id, String prompt, Predicate<String> check) {
            this(id, List.of(prompt), check, false);
        }

        private SmokeCase(String id, List<String> prompts, Predicate<String> check) {
            this(id, prompts, check, false);
        }
    }

    private record Options(String owner, String model, Path modelConfig, Path output, int maxTokens,
                           float temperature, boolean enableThinking) {
        private static Options parse(String[] args) {
            String owner = "edwardcapriolo";
            String model = "Qwen3-4B-JQ4";
            Path modelConfig = Path.of("benchmarks/configs/qwen3-4b-jq4.json");
            Path output = Path.of("core/target/thinking-smoke.jsonl");
            int maxTokens = 768;
            float temperature = 0.0f;
            boolean enableThinking = true;
            for (int i = 0; i < args.length; i++) {
                switch (args[i]) {
                    case "--owner" -> owner = args[++i];
                    case "--model" -> model = args[++i];
                    case "--model-config" -> modelConfig = Path.of(args[++i]);
                    case "--output" -> output = Path.of(args[++i]);
                    case "--max-tokens" -> maxTokens = Integer.parseInt(args[++i]);
                    case "--temperature" -> temperature = Float.parseFloat(args[++i]);
                    case "--thinking" -> enableThinking = Boolean.parseBoolean(args[++i]);
                    default -> throw new IllegalArgumentException("unknown argument: " + args[i]);
                }
            }
            return new Options(owner, model, modelConfig, output, maxTokens, temperature, enableThinking);
        }
    }
}
