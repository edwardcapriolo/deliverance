package io.teknek.deliverance.benchmark;

import com.sun.management.OperatingSystemMXBean;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelConfig;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.tensor.AbstractTensor;

import java.io.BufferedWriter;
import java.lang.management.ManagementFactory;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ForkJoinPool;

import io.teknek.deliverance.math.WrappedForkJoinPool;

/** Direct prefill benchmark over deterministic token IDs, excluding tokenizer, sampling, and decode. */
public final class PrefillTokenBenchmark {
    private static final String CSV_HEADER = "label,owner,model,config,tokens,wall_ms,cpu_ms,cpu_util,tok_s";

    private PrefillTokenBenchmark() {
    }

    public static void main(String[] args) throws Exception {
        Options options = Options.parse(args);
        if (options.output != null && options.output.getParent() != null) {
            Files.createDirectories(options.output.getParent());
        }
        InferenceProfiler.setEnabled(options.profileStages);
        ModelFetcher fetcher = new ModelFetcher(options.owner, options.model);
        AutoModelForCausaLm.Builder builder = AutoModelForCausaLm.newBuilder(fetcher)
                .withWorkingMemoryType(options.workingDType)
                .withWorkingQuantType(options.workingQType)
                .withOutputHeadQuantization(options.outputHeadQuantization)
                .withWrappedForkJoinPool(new WrappedForkJoinPool(new ForkJoinPool(options.poolSize)));
        if (options.modelConfig != null) {
            builder.withConfig(AutoModelConfig.fromJson(options.modelConfig));
        }
        try (AbstractModel model = builder.buildLocalTransformerModel()) {
            System.out.printf(Locale.ROOT,
                    "[prefill-token] loaded label=%s model=%s/%s counts=%s pool=%d config=%s%n",
                    options.label, options.owner, options.model, options.tokenCounts, options.poolSize,
                    options.modelConfig);
            for (int tokenCount : options.tokenCounts) {
                runCase(options, model, tokenCount);
            }
        }
    }

    private static void runCase(Options options, AbstractModel model, int tokenCount) throws Exception {
        int[] tokenIds = deterministicTokens(tokenCount, model.getConfig().vocabularySize);
        InferenceProfiler.reset();
        OperatingSystemMXBean os = (OperatingSystemMXBean) ManagementFactory.getOperatingSystemMXBean();
        long cpuStart = os.getProcessCpuTime();
        long wallStart = System.nanoTime();
        try (AbstractTensor output = model.batchForward(tokenIds, 0)) {
            // The returned final hidden state is intentionally unused; this benchmark measures prefill only.
        }
        long wallNanos = System.nanoTime() - wallStart;
        long cpuNanos = os.getProcessCpuTime() - cpuStart;
        double wallMs = wallNanos / 1_000_000.0d;
        double cpuMs = cpuNanos / 1_000_000.0d;
        double tokPerSecond = tokenCount / (wallNanos / 1_000_000_000.0d);
        double cpuUtil = wallNanos == 0 ? 0.0d : cpuNanos / (double) wallNanos;
        System.out.printf(Locale.ROOT,
                "[prefill-token] label=%s model=%s/%s config=%s tokens=%d wall_ms=%.3f cpu_ms=%.3f cpu_util=%.3f tok_s=%.3f%n",
                options.label, options.owner, options.model, options.modelConfig, tokenCount, wallMs, cpuMs,
                cpuUtil, tokPerSecond);
        if (options.profileStages) {
            InferenceProfiler.printSummary("prefill-token label=" + options.label + " tokens=" + tokenCount,
                    options.profileRows);
            InferenceProfiler.printCounters();
        }
        appendCsv(options, tokenCount, wallMs, cpuMs, cpuUtil, tokPerSecond);
    }

    private static int[] deterministicTokens(int tokenCount, int vocabularySize) {
        if (vocabularySize <= 1001) {
            throw new IllegalArgumentException("vocabularySize too small for deterministic benchmark tokens: "
                    + vocabularySize);
        }
        int[] tokens = new int[tokenCount];
        int span = Math.min(10_000, vocabularySize - 1000);
        for (int i = 0; i < tokenCount; i++) {
            tokens[i] = 1000 + (i % span);
        }
        return tokens;
    }

    private static void appendCsv(Options options, int tokenCount, double wallMs, double cpuMs, double cpuUtil,
            double tokPerSecond) throws Exception {
        if (options.output == null) {
            return;
        }
        boolean writeHeader = !Files.exists(options.output) || Files.size(options.output) == 0;
        try (BufferedWriter writer = Files.newBufferedWriter(options.output, StandardCharsets.UTF_8,
                StandardOpenOption.CREATE, StandardOpenOption.APPEND)) {
            if (writeHeader) {
                writer.write(CSV_HEADER);
                writer.newLine();
            }
            writer.write(String.format(Locale.ROOT, "%s,%s,%s,%s,%d,%.3f,%.3f,%.6f,%.3f",
                    csv(options.label), csv(options.owner), csv(options.model),
                    csv(options.modelConfig == null ? "" : options.modelConfig.toString()), tokenCount, wallMs,
                    cpuMs, cpuUtil, tokPerSecond));
            writer.newLine();
        }
    }

    private static String csv(String value) {
        return '"' + value.replace("\"", "\"\"") + '"';
    }

    private record Options(String label, String owner, String model, Path modelConfig, List<Integer> tokenCounts,
            int poolSize, DType workingDType, DType workingQType, DType outputHeadQuantization,
            boolean profileStages, int profileRows, Path output) {
        private static Options parse(String[] args) {
            String label = "prefill";
            String owner = "edwardcapriolo";
            String model = "Qwen3-4B-JQ4";
            Path modelConfig = null;
            List<Integer> tokenCounts = new ArrayList<>(List.of(100, 500));
            int poolSize = 16;
            DType workingDType = DType.F32;
            DType workingQType = DType.I8;
            DType outputHeadQuantization = DType.Q4;
            boolean profileStages = false;
            int profileRows = 30;
            Path output = null;
            for (int i = 0; i < args.length; i++) {
                switch (args[i]) {
                    case "--label" -> label = args[++i];
                    case "--owner" -> owner = args[++i];
                    case "--model" -> model = args[++i];
                    case "--model-config" -> modelConfig = Path.of(args[++i]);
                    case "--token-counts" -> tokenCounts = parseCounts(args[++i]);
                    case "--pool-size" -> poolSize = Integer.parseInt(args[++i]);
                    case "--working-dtype" -> workingDType = DType.valueOf(args[++i]);
                    case "--working-qtype" -> workingQType = DType.valueOf(args[++i]);
                    case "--output-head-quantization" -> outputHeadQuantization = DType.valueOf(args[++i]);
                    case "--profile-stages" -> profileStages = true;
                    case "--profile-rows" -> profileRows = Integer.parseInt(args[++i]);
                    case "--output" -> output = Path.of(args[++i]);
                    default -> throw new IllegalArgumentException("unknown arg " + args[i]
                            + " in " + Arrays.toString(args));
                }
            }
            return new Options(label, owner, model, modelConfig, List.copyOf(tokenCounts), poolSize, workingDType,
                    workingQType, outputHeadQuantization, profileStages, profileRows, output);
        }

        private static List<Integer> parseCounts(String value) {
            return Arrays.stream(value.split(","))
                    .map(String::trim)
                    .filter(s -> !s.isEmpty())
                    .map(Integer::parseInt)
                    .toList();
        }
    }
}
