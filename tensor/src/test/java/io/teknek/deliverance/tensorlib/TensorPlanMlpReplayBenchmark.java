package io.teknek.deliverance.tensorlib;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.TensorLocality;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.operations.MachineSpec;
import io.teknek.deliverance.tensor.operations.PanamaTensorOperations;
import io.teknek.deliverance.tensor.operations.TensorOperations;

import java.io.BufferedWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.concurrent.ForkJoinPool;

public final class TensorPlanMlpReplayBenchmark {
    private TensorPlanMlpReplayBenchmark() {
    }

    public static void main(String[] args) throws Exception {
        Options options = Options.parse(args);
        Files.createDirectories(options.output.getParent());
        Files.createDirectories(options.jsonOutput.getParent());
        MetricRegistry metrics = new MetricRegistry();
        TensorRuntime runtime = options.runtimeMode == TensorRuntimeMode.DISABLED ? null
                : new TensorRuntime(options.runtimeWorkers, options.runtimeMode, new FakeNative(), metrics);
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new ForkJoinPool(options.poolSize));
             BufferedWriter csv = Files.newBufferedWriter(options.output);
             BufferedWriter json = Files.newBufferedWriter(options.jsonOutput)) {
            TensorOperations ops = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE,
                    new ArrayQueueTensorAllocator(metrics), pool);
            csv.write("m,h,i,runtime_mode,baseline_ms,plan_ms,speedup,max_abs,mean_abs,local,remote,unknown\n");
            json.write("{\"cases\":[\n");
            List<String> jsonRows = new ArrayList<>();
            for (int m : options.mValues) {
                Result result = runCase(ops, pool, runtime, m, options.hidden, options.intermediate, options.seed,
                        options.tensorNumaNode);
                TensorRuntime.LocalitySnapshot snapshot = runtime == null
                        ? new TensorRuntime.LocalitySnapshot(0, 0, 0, 0, 0)
                        : runtime.snapshot();
                csv.write("%d,%d,%d,%s,%.3f,%.3f,%.4f,%.8f,%.8f,%d,%d,%d%n".formatted(m,
                        options.hidden, options.intermediate, options.runtimeMode, result.baselineMs, result.planMs,
                        result.speedup(), result.maxAbs, result.meanAbs, snapshot.local(), snapshot.remote(),
                        snapshot.unknown()));
                jsonRows.add("{\"m\":" + m + ",\"hidden\":" + options.hidden + ",\"intermediate\":"
                        + options.intermediate + ",\"runtimeMode\":\"" + options.runtimeMode
                        + "\",\"baselineMs\":" + result.baselineMs + ",\"planMs\":" + result.planMs
                        + ",\"speedup\":" + result.speedup() + ",\"maxAbs\":" + result.maxAbs
                        + ",\"meanAbs\":" + result.meanAbs + ",\"local\":" + snapshot.local()
                        + ",\"remote\":" + snapshot.remote() + ",\"unknown\":" + snapshot.unknown()
                        + ",\"plan\":" + quote(result.plan) + "}");
                System.out.printf("TENSOR_PLAN_MLP_REPLAY m=%d h=%d i=%d runtime_mode=%s baseline_ms=%.3f plan_ms=%.3f speedup=%.4f max_abs=%.8f mean_abs=%.8f first_diff=%s local=%d remote=%d unknown=%d%n",
                        m, options.hidden, options.intermediate, options.runtimeMode, result.baselineMs,
                        result.planMs, result.speedup(), result.maxAbs, result.meanAbs, result.firstDiff,
                        snapshot.local(), snapshot.remote(), snapshot.unknown());
            }
            json.write(String.join(",\n", jsonRows));
            json.write("\n]}\n");
        } finally {
            if (runtime != null) {
                runtime.close();
            }
        }
    }

    private static Result runCase(TensorOperations ops, WrappedForkJoinPool pool, TensorRuntime runtime, int m, int h,
            int i, int seed, int tensorNumaNode) {
        try (AbstractTensor input = deterministic(m, h, seed);
             AbstractTensor gateW = deterministic(i, h, seed + 1);
             AbstractTensor upW = deterministic(i, h, seed + 2);
             AbstractTensor downW = deterministic(h, i, seed + 3)) {
            attachLocality(input, tensorNumaNode);

            long start = System.nanoTime();
            AbstractTensor baseline = baseline(ops, input, gateW, upW, downW);
            double baselineMs = elapsedMs(start);

            TensorPlan plan = new TensorPlan(ops, pool, null, runtime);
            TensorPlan.Tensor inputNode = plan.input("input", input);
            TensorPlan.ImmutableTensor gateWeight = plan.immutable("gateWeight", gateW);
            TensorPlan.ImmutableTensor upWeight = plan.immutable("upWeight", upW);
            TensorPlan.ImmutableTensor downWeight = plan.immutable("downWeight", downW);
            TensorPlan.Tensor gate = inputNode.batchDot(gateWeight).as("gate");
            TensorPlan.Tensor up = inputNode.batchDot(upWeight).as("up");
            TensorPlan.Tensor hidden = plan.fuse("hidden", gate.shape())
                    .read("gate", gate)
                    .read("up", up)
                    .map("hidden = silu(gate)", TensorPlan.TensorOp.SILU_WRITE,
                            (ctx, offset, length) -> siluWrite(ctx.tensor("gate"), ctx.tensor("hidden"), offset, length))
                    .map("hidden *= up", TensorPlan.TensorOp.MUL_IN_PLACE,
                            (ctx, offset, length) -> multiplyInPlace(ctx.tensor("hidden"), ctx.tensor("up"), offset, length))
                    .tensor();
            TensorPlan.Tensor output = hidden
                    .batchDot(downWeight).as("output");
            String ascii = output.plan();
            start = System.nanoTime();
            AbstractTensor planned = output.materialize();
            double planMs = elapsedMs(start);
            Diff diff = diff(baseline, planned);
            baseline.close();
            planned.close();
            return new Result(baselineMs, planMs, diff.maxAbs, diff.meanAbs, diff.firstDiff, ascii);
        }
    }

    private static AbstractTensor baseline(TensorOperations ops, AbstractTensor input, AbstractTensor gateW,
            AbstractTensor upW, AbstractTensor downW) {
        int m = (int) input.shape().first();
        int h = (int) input.shape().last();
        int i = (int) gateW.shape().first();
        AbstractTensor gate = new FloatBufferTensor(m, i);
        AbstractTensor up = new FloatBufferTensor(m, i);
        AbstractTensor hidden = null;
        try {
            ops.dotProductChunk(gate, input, gateW, 0, h, 0, i);
            ops.dotProductChunk(up, input, upW, 0, h, 0, i);
            for (int row = 0; row < m; row++) {
                for (int col = 0; col < i; col++) {
                    gate.set(ActivationFunction.eval(ActivationFunction.Type.SILU, gate.get(row, col)) * up.get(row, col),
                            row, col);
                }
            }
            hidden = gate;
            AbstractTensor output = new FloatBufferTensor(m, (int) downW.shape().first());
            ops.dotProductChunk(output, hidden, downW, 0, i, 0, (int) downW.shape().first());
            return output;
        } finally {
            if (hidden != gate) {
                gate.close();
            }
            up.close();
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

    private static void attachLocality(AbstractTensor tensor, int numaNode) {
        tensor.setLocality(new TensorLocality(tensor.getMemorySegment().address(), tensor.getMemorySegment().byteSize(),
                numaNode, List.of(numaNode), System.currentTimeMillis(), "fake-replay"));
    }

    private static void siluWrite(AbstractTensor input, AbstractTensor output, long offset, long length) {
        int cols = (int) input.shape().last();
        for (long index = offset; index < offset + length; index++) {
            int row = (int) (index / cols);
            int col = (int) (index % cols);
            output.set(ActivationFunction.eval(ActivationFunction.Type.SILU, input.get(row, col)), row, col);
        }
    }

    private static void multiplyInPlace(AbstractTensor lhs, AbstractTensor rhs, long offset, long length) {
        int cols = (int) lhs.shape().last();
        for (long index = offset; index < offset + length; index++) {
            int row = (int) (index / cols);
            int col = (int) (index % cols);
            lhs.set(lhs.get(row, col) * rhs.get(row, col), row, col);
        }
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

    private record Result(double baselineMs, double planMs, float maxAbs, double meanAbs, String firstDiff, String plan) {
        double speedup() {
            return baselineMs / planMs;
        }
    }

    private record Diff(float maxAbs, double meanAbs, String firstDiff) {
    }

    private record Options(Path output, Path jsonOutput, int poolSize, int runtimeWorkers, TensorRuntimeMode runtimeMode,
                           int tensorNumaNode, int hidden, int intermediate, List<Integer> mValues, int seed) {
        static Options parse(String[] args) {
            Path output = Path.of("target/tensor-plan-mlp-replay.csv");
            Path jsonOutput = Path.of("target/tensor-plan-mlp-replay.json");
            int poolSize = Math.max(1, Runtime.getRuntime().availableProcessors() / 2);
            int runtimeWorkers = poolSize;
            TensorRuntimeMode runtimeMode = TensorRuntimeMode.DISABLED;
            int tensorNumaNode = 1;
            int hidden = Integer.getInteger("deliverance.tensorplan.hidden", 512);
            int intermediate = Integer.getInteger("deliverance.tensorplan.intermediate", 1536);
            int seed = Integer.getInteger("deliverance.tensorplan.seed", 7);
            List<Integer> mValues = new ArrayList<>(List.of(32, 64, 128));
            for (int idx = 0; idx < args.length; idx++) {
                switch (args[idx]) {
                    case "--output" -> output = Path.of(args[++idx]);
                    case "--json-output" -> jsonOutput = Path.of(args[++idx]);
                    case "--pool-size" -> poolSize = Integer.parseInt(args[++idx]);
                    case "--runtime-workers" -> runtimeWorkers = Integer.parseInt(args[++idx]);
                    case "--runtime-mode" -> runtimeMode = TensorRuntimeMode.valueOf(args[++idx].trim().toUpperCase());
                    case "--tensor-numa-node" -> tensorNumaNode = Integer.parseInt(args[++idx]);
                    case "--hidden" -> hidden = Integer.parseInt(args[++idx]);
                    case "--intermediate" -> intermediate = Integer.parseInt(args[++idx]);
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
            return new Options(output, jsonOutput, poolSize, runtimeWorkers, runtimeMode, tensorNumaNode, hidden,
                    intermediate, List.copyOf(mValues), seed);
        }
    }

    private static final class FakeNative implements TensorRuntimeNative {
        @Override
        public boolean available() {
            return true;
        }

        @Override
        public String reason() {
            return "fake two-node topology";
        }

        @Override
        public Optional<TensorLocality> localityOf(AbstractTensor tensor) {
            return tensor.locality();
        }

        @Override
        public OptionalInt currentCpu() {
            return OptionalInt.empty();
        }

        @Override
        public OptionalInt currentNumaNode() {
            return OptionalInt.empty();
        }

        @Override
        public OptionalInt numaNodeOfCpu(int cpu) {
            return OptionalInt.of(cpu % 2);
        }

        @Override
        public boolean pinCurrentThread(int cpu) {
            return true;
        }
    }
}
