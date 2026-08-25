package io.teknek.deliverance.tensorlib;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.operations.NaiveTensorOperations;

import java.io.BufferedWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

public final class VariableMlpFusionReplayBenchmark {
    private VariableMlpFusionReplayBenchmark() {
    }

    public static void main(String[] args) throws Exception {
        Options options = Options.parse(args);
        Files.createDirectories(options.output.getParent());
        Files.createDirectories(options.jsonOutput.getParent());
        MetricRegistry metrics = new MetricRegistry();
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new ForkJoinPool(options.poolSize));
             BufferedWriter csv = Files.newBufferedWriter(options.output);
             BufferedWriter json = Files.newBufferedWriter(options.jsonOutput)) {
            csv.write("m,hidden,baseline_ms,fixed_plan_ms,intstream_plan_ms,fixed_speedup,intstream_speedup,fixed_max_abs,intstream_max_abs,fixed_mean_abs,intstream_mean_abs\n");
            json.write("{\"cases\":[\n");
            List<String> jsonRows = new ArrayList<>();
            for (int m : options.mValues) {
                Result result = runCase(pool, metrics, m, options.hidden, options.seed);
                csv.write("%d,%d,%.3f,%.3f,%.3f,%.4f,%.4f,%.8f,%.8f,%.8f,%.8f%n".formatted(m,
                        options.hidden, result.baselineMs, result.fixedPlanMs, result.intStreamPlanMs,
                        result.fixedSpeedup(), result.intStreamSpeedup(), result.fixedMaxAbs, result.intStreamMaxAbs,
                        result.fixedMeanAbs, result.intStreamMeanAbs));
                jsonRows.add("{\"m\":" + m + ",\"hidden\":" + options.hidden + ",\"baselineMs\":"
                        + result.baselineMs + ",\"fixedPlanMs\":" + result.fixedPlanMs + ",\"intStreamPlanMs\":"
                        + result.intStreamPlanMs + ",\"fixedSpeedup\":" + result.fixedSpeedup()
                        + ",\"intStreamSpeedup\":" + result.intStreamSpeedup() + ",\"fixedMaxAbs\":"
                        + result.fixedMaxAbs + ",\"intStreamMaxAbs\":" + result.intStreamMaxAbs
                        + ",\"fixedMeanAbs\":" + result.fixedMeanAbs + ",\"intStreamMeanAbs\":"
                        + result.intStreamMeanAbs + ",\"fixedPlan\":" + quote(result.fixedPlan)
                        + ",\"intStreamPlan\":" + quote(result.intStreamPlan) + "}");
                System.out.printf("VARIABLE_MLP_FUSION_REPLAY m=%d hidden=%d baseline_ms=%.3f fixed_plan_ms=%.3f intstream_plan_ms=%.3f fixed_speedup=%.4f intstream_speedup=%.4f fixed_max_abs=%.8f intstream_max_abs=%.8f fixed_first_diff=%s intstream_first_diff=%s%n",
                        m, options.hidden, result.baselineMs, result.fixedPlanMs, result.intStreamPlanMs,
                        result.fixedSpeedup(), result.intStreamSpeedup(), result.fixedMaxAbs, result.intStreamMaxAbs,
                        result.fixedFirstDiff, result.intStreamFirstDiff);
            }
            json.write(String.join(",\n", jsonRows));
            json.write("\n]}\n");
        }
    }

    private static Result runCase(WrappedForkJoinPool pool, MetricRegistry metrics, int m, int hidden, int seed) {
        try (AbstractTensor baselineGate = deterministic(m, hidden, seed);
             AbstractTensor fixedPlanGate = deterministic(m, hidden, seed);
             AbstractTensor intStreamPlanGate = deterministic(m, hidden, seed);
             AbstractTensor up = deterministic(m, hidden, seed + 1)) {

            long start = System.nanoTime();
            baseline(baselineGate, up);
            double baselineMs = elapsedMs(start);

            TensorPlan fixedPlan = new TensorPlan(new NaiveTensorOperations(), pool, metrics);
            TensorPlan.Tensor fixedOutput = fixedPlan.mutable("gate", fixedPlanGate)
                    .activate(ActivationFunction.Type.SILU)
                    .multiply(fixedPlan.input("up", up))
                    .timer("variablemlpblock.fixed_multiply");
            String fixedAscii = fixedOutput.plan();
            start = System.nanoTime();
            fixedOutput.materialize();
            double fixedPlanMs = elapsedMs(start);

            TensorPlan intStreamPlan = new TensorPlan(new NaiveTensorOperations(), pool, metrics);
            TensorPlan.Tensor intStreamOutput = intStreamPlan.fuseColumnsIntStream("gate", intStreamPlanGate.shape())
                    .write("gate", intStreamPlan.mutable("gate", intStreamPlanGate))
                    .read("up", intStreamPlan.input("up", up))
                    .map("gate = silu(gate) * up", TensorPlan.TensorOp.ACTIVATION_MUL_IN_PLACE,
                            (ctx, offset, length) -> baselineColumn(ctx.tensor("gate"), ctx.tensor("up"), (int) offset))
                    .tensor()
                    .timer("variablemlpblock.intstream_multiply");
            String intStreamAscii = intStreamOutput.plan();
            start = System.nanoTime();
            intStreamOutput.materialize();
            double intStreamPlanMs = elapsedMs(start);

            Diff fixedDiff = diff(baselineGate, fixedPlanGate);
            Diff intStreamDiff = diff(baselineGate, intStreamPlanGate);
            return new Result(baselineMs, fixedPlanMs, intStreamPlanMs, fixedDiff.maxAbs, intStreamDiff.maxAbs,
                    fixedDiff.meanAbs, intStreamDiff.meanAbs, fixedDiff.firstDiff, intStreamDiff.firstDiff,
                    fixedAscii, intStreamAscii);
        }
    }

    private static void baseline(AbstractTensor gate, AbstractTensor up) {
        int batchSize = (int) gate.shape().first();
        int hidden = (int) gate.shape().last();
        IntStream.range(0, hidden).parallel().forEach(i -> {
            for (int j = 0; j < batchSize; j++) {
                float activated = ActivationFunction.eval(ActivationFunction.Type.SILU, gate.get(j, i));
                gate.set(activated * up.get(j, i), j, i);
            }
        });
    }

    private static void baselineColumn(AbstractTensor gate, AbstractTensor up, int column) {
        int batchSize = (int) gate.shape().first();
        for (int row = 0; row < batchSize; row++) {
            float activated = ActivationFunction.eval(ActivationFunction.Type.SILU, gate.get(row, column));
            gate.set(activated * up.get(row, column), row, column);
        }
    }

    private static AbstractTensor deterministic(int rows, int cols, int seed) {
        FloatBufferTensor tensor = new FloatBufferTensor(TensorShape.of(rows, cols));
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                tensor.set((((row * 31 + col * 17 + seed) % 29 - 14) / 14.0f) * 0.125f, row, col);
            }
        }
        return tensor;
    }

    private static Diff diff(AbstractTensor a, AbstractTensor b) {
        double sum = 0;
        float max = 0;
        long count = a.size();
        for (int row = 0; row < a.shape().first(); row++) {
            for (int col = 0; col < a.shape().last(); col++) {
                float d = Math.abs(a.get(row, col) - b.get(row, col));
                max = Math.max(max, d);
                sum += d;
            }
        }
        return new Diff(max, sum / count, firstDiff(a, b));
    }

    private static String firstDiff(AbstractTensor a, AbstractTensor b) {
        for (int row = 0; row < a.shape().first(); row++) {
            for (int col = 0; col < a.shape().last(); col++) {
                float av = a.get(row, col);
                float bv = b.get(row, col);
                if (Math.abs(av - bv) > 1.0e-6f) {
                    return "row=" + row + " col=" + col + " baseline=" + av + " plan=" + bv;
                }
            }
        }
        return "none";
    }

    private static double elapsedMs(long start) {
        return (System.nanoTime() - start) / 1_000_000.0;
    }

    private static String quote(String value) {
        return "\"" + value.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n") + "\"";
    }

    private record Result(double baselineMs, double fixedPlanMs, double intStreamPlanMs, float fixedMaxAbs,
                          float intStreamMaxAbs, double fixedMeanAbs, double intStreamMeanAbs, String fixedFirstDiff,
                          String intStreamFirstDiff, String fixedPlan, String intStreamPlan) {
        double fixedSpeedup() {
            return baselineMs / fixedPlanMs;
        }

        double intStreamSpeedup() {
            return baselineMs / intStreamPlanMs;
        }
    }

    private record Diff(float maxAbs, double meanAbs, String firstDiff) {
    }

    private record Options(Path output, Path jsonOutput, int poolSize, int hidden, List<Integer> mValues, int seed) {
        static Options parse(String[] args) {
            Path output = Path.of("target/variable-mlp-fusion-replay.csv");
            Path jsonOutput = Path.of("target/variable-mlp-fusion-replay.json");
            int poolSize = Math.max(1, Runtime.getRuntime().availableProcessors() / 2);
            int hidden = Integer.getInteger("deliverance.variablemlp.hidden", 3072);
            int seed = Integer.getInteger("deliverance.variablemlp.seed", 7);
            List<Integer> mValues = new ArrayList<>(List.of(1, 32, 128, 256, 403));
            for (int idx = 0; idx < args.length; idx++) {
                switch (args[idx]) {
                    case "--output" -> output = Path.of(args[++idx]);
                    case "--json-output" -> jsonOutput = Path.of(args[++idx]);
                    case "--pool-size" -> poolSize = Integer.parseInt(args[++idx]);
                    case "--hidden" -> hidden = Integer.parseInt(args[++idx]);
                    case "--m-values" -> {
                        mValues = new ArrayList<>();
                        for (String part : args[++idx].split(",")) {
                            mValues.add(Integer.parseInt(part.trim()));
                        }
                    }
                    case "--seed" -> seed = Integer.parseInt(args[++idx]);
                    default -> throw new IllegalArgumentException("Unknown argument " + args[idx]);
                }
            }
            return new Options(output, jsonOutput, poolSize, hidden, List.copyOf(mValues), seed);
        }
    }
}
