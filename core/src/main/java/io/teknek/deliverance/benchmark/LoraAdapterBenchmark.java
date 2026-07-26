package io.teknek.deliverance.benchmark;

import com.fasterxml.jackson.databind.node.ObjectNode;
import io.teknek.deliverance.JsonUtils;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.model.DoNothingGenerateEvent;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.safetensors.fetch.LoraAdapterModelFetcher;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;

import java.io.BufferedWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.UUID;

/**
 * Compares an un-adapted base model against the same base model with a LoRA adapter merged in
 * (Phase 1 "merge-at-load", {@link io.teknek.deliverance.safetensors.MergingWeightLoader}), added
 * per the maintainer's PR feedback that new features should ship with a profiled, example-
 * generating benchmark -- see {@code StepPlans/deliverance_lora_step3_merging_weightloader_plan_v1.md}
 * Section 6.
 *
 * <p>Structural template is {@link ThinkingSmokeBenchmark} (one model at a time, hand-rolled CLI,
 * small fixed prompt list), not {@link InferenceBenchmark} (multi-engine, tensor-parallel-aware) --
 * but the CSV column set is deliberately identical to {@link InferenceBenchmark}'s documented
 * columns, plus a {@code variant} column distinguishing {@code base} from {@code lora-adapted}
 * rows, so the same row can be diffed/filtered across both variants to surface the adapter's
 * overhead directly.</p>
 */
public final class LoraAdapterBenchmark {
    private static final String CSV_HEADER = String.join(",",
            "timestamp",
            "variant",
            "model",
            "case_id",
            "category",
            "prompt_chars",
            "prompt_tokens",
            "generated_tokens",
            "total_ms",
            "generation_ms",
            "tokens_per_second",
            "response_chars",
            "finish_reason");

    private LoraAdapterBenchmark() {
    }

    public static void main(String[] args) throws Exception {
        Options options = Options.parse(args);
        if (options.output.getParent() != null) {
            Files.createDirectories(options.output.getParent());
        }
        if (options.jsonlOutput.getParent() != null) {
            Files.createDirectories(options.jsonlOutput.getParent());
        }
        InferenceProfiler.setEnabled(options.profileStages);

        try (BufferedWriter csv = Files.newBufferedWriter(options.output, StandardCharsets.UTF_8);
                BufferedWriter jsonl = Files.newBufferedWriter(options.jsonlOutput, StandardCharsets.UTF_8)) {
            csv.write(CSV_HEADER);
            csv.newLine();
            csv.flush();
            System.out.println("writing benchmark results to " + options.output.toAbsolutePath());
            System.out.println("writing benchmark transcripts to " + options.jsonlOutput.toAbsolutePath());

            ModelFetcher fetcher = new ModelFetcher(options.owner, options.model);
            String modelName = options.owner + "/" + options.model;

            System.out.println("[lora-adapter-benchmark] loading base model " + modelName);
            try (AbstractModel base = AutoModelForCausaLm.newBuilder(fetcher)
                    .withWrappedForkJoinPool(new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores()))
                    .buildLocalTransformerModel()) {
                runVariant("base", modelName, base, options, csv, jsonl);
            }

            String adapterName = options.adapterOwner + "/" + options.adapterModel;
            System.out.println("[lora-adapter-benchmark] loading base model " + modelName
                    + " with LoRA adapter " + adapterName + " merged in");
            try (AbstractModel adapted = AutoModelForCausaLm.newBuilder(fetcher)
                    .withWrappedForkJoinPool(new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores()))
                    .withLoraAdapter(new LoraAdapterModelFetcher(options.adapterOwner, options.adapterModel))
                    .buildLocalTransformerModel()) {
                runVariant("lora-adapted", modelName + "+" + adapterName, adapted, options, csv, jsonl);
            }
        }
        System.out.println("wrote benchmark results to " + options.output.toAbsolutePath());
    }

    private static void runVariant(String variant, String modelLabel, AbstractModel model, Options options,
            BufferedWriter csv, BufferedWriter jsonl) throws Exception {
        for (PromptCase promptCase : promptCases()) {
            InferenceProfiler.reset();
            PromptSupport.Builder promptBuilder = model.promptSupport().orElseThrow().builder()
                    .addUserMessage(promptCase.prompt);
            var promptContext = promptBuilder.build();

            long start = System.nanoTime();
            Response response = model.generate(UUID.randomUUID(), promptContext,
                    new GeneratorParameters().withTemperature(options.temperature).withMaxTokens(options.maxTokens),
                    new DoNothingGenerateEvent());
            double totalMs = (System.nanoTime() - start) / 1_000_000.0;

            double timeToFirstTokenMs = response.timeToFirstTokenMs;
            double generationMs = Math.max(0.0, totalMs - timeToFirstTokenMs);
            long decodeTokens = Math.max(0, response.generatedTokens.size() - 1L);
            double tokensPerSecond = generationMs == 0.0 ? 0.0 : decodeTokens / (generationMs / 1000.0);

            writeCsvRow(csv, variant, modelLabel, promptCase, promptContext.getPrompt().length(),
                    response, totalMs, generationMs, tokensPerSecond);
            writeJsonlRow(jsonl, variant, promptCase, promptContext.getPrompt(), response, totalMs, timeToFirstTokenMs);

            System.out.printf(Locale.ROOT,
                    "[lora-adapter-benchmark] variant=%s case=%s generated=%d total_ms=%.1f tok_s=%.2f finish=%s%n",
                    variant, promptCase.id, response.generatedTokens.size(), totalMs, tokensPerSecond,
                    response.finishReason == null ? "" : response.finishReason.name());
            InferenceProfiler.printSummary("variant=" + variant + " case=" + promptCase.id, 20);
        }
    }

    private static void writeCsvRow(BufferedWriter csv, String variant, String modelLabel, PromptCase promptCase,
            int promptChars, Response response, double totalMs, double generationMs, double tokensPerSecond)
            throws Exception {
        csv.write(String.join(",",
                csv(Instant.now().toString()),
                csv(variant),
                csv(modelLabel),
                csv(promptCase.id),
                csv(promptCase.category),
                Integer.toString(promptChars),
                Long.toString(response.promptTokens),
                Long.toString(response.generatedTokens.size()),
                String.format(Locale.ROOT, "%.3f", totalMs),
                String.format(Locale.ROOT, "%.3f", generationMs),
                String.format(Locale.ROOT, "%.3f", tokensPerSecond),
                Integer.toString(response.responseText.length()),
                csv(response.finishReason == null ? "" : response.finishReason.name())));
        csv.newLine();
        csv.flush();
    }

    private static void writeJsonlRow(BufferedWriter jsonl, String variant, PromptCase promptCase, String prompt,
            Response response, double totalMs, double timeToFirstTokenMs) throws Exception {
        ObjectNode row = JsonUtils.om.createObjectNode();
        row.put("timestamp", Instant.now().toString());
        row.put("case_id", promptCase.id);
        row.put("variant", variant);
        row.put("prompt", prompt);
        row.put("response", response.responseText);
        row.put("generated_tokens", response.generatedTokens.size());
        row.put("time_ms", totalMs);
        row.put("time_to_first_token_ms", timeToFirstTokenMs);
        row.put("finish_reason", response.finishReason == null ? "" : response.finishReason.name());
        jsonl.write(JsonUtils.om.writeValueAsString(row));
        jsonl.newLine();
        jsonl.flush();
    }

    private static String csv(String value) {
        if (value == null) {
            return "";
        }
        String escaped = value.replace("\"", "\"\"");
        return "\"" + escaped + "\"";
    }

    private static List<PromptCase> promptCases() {
        return List.of(
                new PromptCase("greeting", "chat", "Hello! Who are you and what can you help me with?"),
                new PromptCase("capital", "knowledge", "What is the capital of France? Answer in one word."),
                new PromptCase("math", "math", "What is 17 times 6? Show your work briefly."),
                new PromptCase("summary", "writing",
                        "Summarize in two sentences: the water cycle moves water between the ocean, atmosphere, and "
                                + "land through evaporation, condensation, and precipitation."),
                new PromptCase("code", "coding", "Write a Python function that returns the nth Fibonacci number."),
                new PromptCase("opinion", "reasoning", "What are the tradeoffs between electric and gas cars? Give two of each."));
    }

    private record PromptCase(String id, String category, String prompt) {
    }

    private record Options(String owner, String model, String adapterOwner, String adapterModel, Path output,
            Path jsonlOutput, int maxTokens, float temperature, boolean profileStages) {
        private static Options parse(String[] args) {
            String owner = "unsloth";
            String model = "Llama-3.2-1B-Instruct";
            String adapterOwner = "bunnycore";
            String adapterModel = "Llama-3.2-1b-chatml-lora_model";
            Path output = Path.of("core/target/lora-adapter-benchmark.csv");
            Path jsonlOutput = Path.of("core/target/lora-adapter-benchmark.jsonl");
            int maxTokens = 128;
            float temperature = 0.0f;
            boolean profileStages = false;
            for (int i = 0; i < args.length; i++) {
                switch (args[i]) {
                    case "--owner" -> owner = args[++i];
                    case "--model" -> model = args[++i];
                    case "--adapter-owner" -> adapterOwner = args[++i];
                    case "--adapter-model" -> adapterModel = args[++i];
                    case "--output" -> output = Path.of(args[++i]);
                    case "--jsonl-output" -> jsonlOutput = Path.of(args[++i]);
                    case "--max-tokens" -> maxTokens = Integer.parseInt(args[++i]);
                    case "--temperature" -> temperature = Float.parseFloat(args[++i]);
                    case "--profile-stages" -> profileStages = true;
                    default -> throw new IllegalArgumentException("unknown argument: " + args[i]);
                }
            }
            return new Options(owner, model, adapterOwner, adapterModel, output, jsonlOutput, maxTokens, temperature,
                    profileStages);
        }
    }
}
