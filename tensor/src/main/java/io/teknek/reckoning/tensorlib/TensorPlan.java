package io.teknek.reckoning.tensorlib;

import com.google.common.base.Preconditions;
import io.dropwizard.metrics5.MetricRegistry;
import io.dropwizard.metrics5.Timer;
import io.teknek.reckoning.DType;
import io.teknek.reckoning.math.ActivationFunction;
import io.teknek.reckoning.math.WrappedForkJoinPool;
import io.teknek.reckoning.tensor.TensorShape;
import io.teknek.reckoning.tensor2.BatchDotProduct;
import io.teknek.reckoning.tensor2.GemmKernel;
import io.teknek.reckoning.tensor2.Lighter;
import io.teknek.reckoning.tensor2.Scale;
import io.teknek.reckoning.tensor2.TensorRef;
import io.teknek.reckoning.tensor2.Threading;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Small lazy tensor workflow for experimenting with logical tensor plans and local fusion rules.
 *
 * <p>External tensors are introduced with {@link #input(String, TensorRef)}. Operations build a logical graph and
 * execute only when {@link Tensor#materializeRef()} is called.</p>
 */
public final class TensorPlan {
    private static final Logger LOGGER = LoggerFactory.getLogger(TensorPlan.class);
    public enum RunMode {
        DEFAULT,
        CALLER_THREAD
    }

    private final WrappedForkJoinPool pool;
    private final MetricRegistry metricRegistry;
    private final TensorRuntime runtime;
    private final Lighter lighter;
    private final String ownerClass;
    private final TensorPlanAdaptiveSplitTuner adaptiveSplitTuner;
    private RunMode runMode = RunMode.DEFAULT;

    public TensorPlan(WrappedForkJoinPool pool) {
        this(pool, null);
    }

    public TensorPlan(WrappedForkJoinPool pool, MetricRegistry metricRegistry) {
        this(pool, metricRegistry, null);
    }

    public TensorPlan(WrappedForkJoinPool pool, MetricRegistry metricRegistry,
            Object owner) {
        this(pool, metricRegistry, owner, null);
    }

    public TensorPlan(WrappedForkJoinPool pool, MetricRegistry metricRegistry,
            TensorRuntime runtime) {
        this(pool, metricRegistry, null, runtime);
    }

    public TensorPlan(WrappedForkJoinPool pool, MetricRegistry metricRegistry,
            Object owner, TensorRuntime runtime) {
        this(pool, metricRegistry, owner, runtime, new Lighter(metricRegistry == null ? new MetricRegistry()
                : metricRegistry));
    }

    public TensorPlan(WrappedForkJoinPool pool, MetricRegistry metricRegistry,
            Object owner, TensorRuntime runtime, Lighter lighter) {
        this.pool = Objects.requireNonNull(pool, "pool");
        this.metricRegistry = metricRegistry;
        this.runtime = runtime;
        this.lighter = Objects.requireNonNull(lighter, "lighter");
        this.ownerClass = owner == null ? "UNKNOWN" : owner.getClass().getSimpleName();
        this.adaptiveSplitTuner = owner instanceof TensorPlanAdaptiveSplitTuner tuner ? tuner : null;
    }


    public String ownerClass() {
        return ownerClass;
    }

    public Tensor input(String name, TensorRef tensor) {
        Objects.requireNonNull(tensor, "tensor");
        return new Tensor(new RefInputNode(name, tensor, false));
    }

    public Tensor input(String name, Tensor upstream, TensorRef materialized) {
        Objects.requireNonNull(upstream, "upstream");
        Objects.requireNonNull(materialized, "materialized");
        return new Tensor(new RefImportedInputNode(name, upstream.node, materialized));
    }

    public ImmutableTensor immutable(String name, TensorRef tensor) {
        Objects.requireNonNull(tensor, "tensor");
        return new ImmutableTensor(new RefInputNode(name, tensor, false));
    }

    public Tensor input(String name, ImmutableTensor upstream, TensorRef materialized) {
        Objects.requireNonNull(upstream, "upstream");
        Objects.requireNonNull(materialized, "materialized");
        return new Tensor(new RefImportedInputNode(name, upstream.node, materialized));
    }

    /**
     * Creates a fused node with an explicit iteration/output shape.
     *
     * <p>The shape is both the default output shape when the fuse creates a new tensor and the iteration domain used by
     * {@link FusedRefMap}. For {@link FusedExecution#FIXED_POOL_LINEAR}, materialization splits {@code shape.size()} into
     * linear chunks and passes each chunk as {@code offset}/{@code length} to every fused map. Callers that want row-chunk
     * semantics should use a one-dimensional shape such as {@code TensorShape.of(rowCount)} and interpret the map offset
     * as a row start.</p>
     */
    public FusedBuilder fuse(String name, TensorShape shape) {
        return new FusedBuilder(name, shape, FusedExecution.FIXED_POOL_LINEAR);
    }

    public FusedBuilder fuseColumnsIntStream(String name, TensorShape shape) {
        return new FusedBuilder(name, shape, FusedExecution.INT_STREAM_COLUMNS);
    }

    public FusedBuilder fuseRowsIntStream(String name, TensorShape shape) {
        return new FusedBuilder(name, shape, FusedExecution.INT_STREAM_ROWS);
    }

    public AlternateDitBuilder alternateDit(String planName, String candidateName, Runnable operation) {
        return new AlternateDitBuilder(planName, candidateName, operation);
    }

    public ChunkedBuilder chunked(String operation, long offset, long length) {
        return new ChunkedBuilder(operation, offset, length);
    }

    public TensorPlan forcedRunMode(RunMode runMode) {
        this.runMode = Objects.requireNonNull(runMode, "runMode");
        return this;
    }

    public RunMode runMode() {
        return runMode;
    }

    public final class ChunkedBuilder {
        private final String operation;
        private final long offset;
        private final long length;
        private Integer splitCount;

        private ChunkedBuilder(String operation, long offset, long length) {
            if (length < 0) {
                throw new IllegalArgumentException("length must be >= 0");
            }
            this.operation = Objects.requireNonNull(operation, "operation");
            this.offset = offset;
            this.length = length;
        }

        public ChunkedBuilder splitCount(int splitCount) {
            if (splitCount < 1) {
                throw new IllegalArgumentException("splitCount must be >= 1");
            }
            this.splitCount = splitCount;
            return this;
        }

        public void run(Chunk action) {
            Objects.requireNonNull(action, "action");
            int splitsCount = splitCount == null ? Math.max(1, pool.getCoreCount()) : splitCount;
            List<TensorSplit> splits = TensorLib.calculateTSplits(offset, length, splitsCount);
            if (useTensorRuntime()) {
                runtime.runChunks(operation, Math.toIntExact(offset), Math.toIntExact(length), splits.size(),
                        Optional.empty(), action::run);
                return;
            }
            List<ForkJoinTask<?>> tasks = new ArrayList<>();
            for (TensorSplit split : splits) {
                tasks.add(pool.getUnderlying().submit(() -> action.run(split.offset(), split.length())));
            }
            tasks.forEach(ForkJoinTask::join);
        }
    }

    @FunctionalInterface
    public interface Chunk {
        void run(long offset, long length);
    }

    public final class AlternateDitBuilder {
        private final String planName;
        private final Map<String, Runnable> candidates = new LinkedHashMap<>();

        private AlternateDitBuilder(String planName, String candidateName, Runnable operation) {
            this.planName = Objects.requireNonNull(planName, "planName");
            alternate(candidateName, operation);
        }

        public AlternateDitBuilder alternate(String candidateName, Runnable operation) {
            candidates.put(Objects.requireNonNull(candidateName, "candidateName"),
                    Objects.requireNonNull(operation, "operation"));
            return this;
        }

        public void run() {
            if (candidates.isEmpty()) {
                throw new IllegalStateException("alternateDit requires at least one candidate");
            }
            String candidate = adaptiveSplitTuner == null
                    ? candidates.keySet().iterator().next()
                    : adaptiveSplitTuner.chooseAlternate(planName, List.copyOf(candidates.keySet()));
            Runnable operation = candidates.get(candidate);
            if (operation == null) {
                candidate = candidates.keySet().iterator().next();
                operation = candidates.get(candidate);
            }
            long start = System.nanoTime();
            try {
                operation.run();
            } finally {
                if (adaptiveSplitTuner != null) {
                    adaptiveSplitTuner.observeAlternate(planName, candidate, System.nanoTime() - start);
                }
            }
        }
    }

    public final class Tensor {
        private final Node node;

        private Tensor(Node node) {
            this.node = node;
        }

        public Tensor batchDot(Tensor weight) {
            return new Tensor(new BatchDotNode(this.node, weight.node));
        }

        public Tensor batchDot(ImmutableTensor weight) {
            return new Tensor(new BatchDotNode(this.node, weight.node));
        }

        public Tensor mlp(ImmutableTensor gateWeight, ImmutableTensor upWeight, ImmutableTensor downWeight,
                ActivationFunction.Type activation, DType quantizedType) {
            return new Tensor(new MlpNode(this.node, gateWeight.node, upWeight.node, downWeight.node, activation,
                    quantizedType));
        }

        public Tensor activate(ActivationFunction.Type activation) {
            return new Tensor(new ActivationNode(this.node, activation));
        }

        public Tensor multiply(Tensor other) {
            return new Tensor(new MultiplyNode(this.node, other.node));
        }

        public Tensor add(Tensor other) {
            return new Tensor(new AddNode(this.node, other.node));
        }

        public Tensor scale(float factor) {
            return new Tensor(new ScaleNode(this.node, factor));
        }

        public Tensor quantize(DType dtype) {
            return new Tensor(new QuantizeNode(this.node, dtype));
        }

        public Tensor as(String name) {
            return new Tensor(new NamedNode(name, this.node));
        }

        public Tensor timer(String metricName) {
            return new Tensor(new TimedNode(metricName, this.node));
        }

        /**
         * Opts this tensor into model-scoped adaptive split tuning during materialization.
         *
         * <p>This currently applies to fused nodes. TensorPlan starts with the node's normal split policy, then asks the
         * owning model's {@link TensorPlanAdaptiveSplitTuner} to try split counts in the supplied inclusive range and reuse
         * the fastest observed split on later materializations of the same named shape.</p>
         */
        public Tensor justInTime(int minSplit, int maxSplit) {
            return new Tensor(new JustInTimeNode(this.node, minSplit, maxSplit));
        }

        public TensorShape shape() {
            return node.shape();
        }

        public TensorRef materializeRef() {
            if (LOGGER.isDebugEnabled()) {
                LOGGER.debug("TensorPlan:\n{}", plan());
            }
            return node.evalRef().tensor();
        }

        public String plan() {
            StringBuilder sb = new StringBuilder();
            node.render(sb, "", true);
            return sb.toString();
        }
    }

    public final class ImmutableTensor {
        private final Node node;

        private ImmutableTensor(Node node) {
            this.node = node;
        }

        public String plan() {
            StringBuilder sb = new StringBuilder();
            node.render(sb, "", true);
            return sb.toString();
        }
    }

    public enum TensorOp {
        CUSTOM,
        SILU_WRITE,
        MUL_IN_PLACE,
        ACTIVATION_MUL_IN_PLACE,
        ACTIVATION_SPARSITY_IN_PLACE
    }

    private enum FusedExecution {
        FIXED_POOL_LINEAR,
        INT_STREAM_COLUMNS,
        INT_STREAM_ROWS
    }

    @FunctionalInterface
    public interface FusedRefMap {
        void map(FusedRefContext context, long offset, long length);
    }

    public final class FusedRefContext {
        private final Map<String, TensorRef> tensors;

        private FusedRefContext(Map<String, TensorRef> tensors) {
            this.tensors = tensors;
        }

        public TensorRef tensor(String name) {
            TensorRef tensor = tensors.get(name);
            if (tensor == null) {
                throw new IllegalArgumentException("Unknown fused tensor " + name);
            }
            return tensor;
        }
    }

    public final class FusedBuilder {
        private final String name;
        private final TensorShape shape;
        private final FusedExecution execution;
        private final Map<String, Node> inputs = new LinkedHashMap<>();
        private final List<FusedRefStep> refSteps = new ArrayList<>();
        private String outputInputName;
        private Integer splitCount;

        private FusedBuilder(String name, TensorShape shape, FusedExecution execution) {
            this.name = Objects.requireNonNull(name, "name");
            this.shape = Objects.requireNonNull(shape, "shape");
            this.execution = Objects.requireNonNull(execution, "execution");
        }

        public FusedBuilder read(String name, Tensor tensor) {
            inputs.put(name, tensor.node);
            return this;
        }

        public FusedBuilder read(String name, ImmutableTensor tensor) {
            inputs.put(name, tensor.node);
            return this;
        }

        public FusedBuilder write(String name, Tensor tensor) {
            read(name, tensor);
            outputInputName = name;
            return this;
        }

        public FusedBuilder mapRef(String description, TensorOp op, FusedRefMap map) {
            return mapRef(description, op, null, map);
        }

        public FusedBuilder mapRef(String description, TensorOp op, String metricName, FusedRefMap map) {
            refSteps.add(new FusedRefStep(description, op, metricName, map));
            return this;
        }

        /**
         * Overrides the default fused-node split count.
         *
         * <p>By default, fixed-pool fused nodes split the iteration domain by {@code pool.getCoreCount()}. This override
         * is useful when a shape has a natural model partition, such as attention-head or KV-head row groups, that is more
         * efficient than blindly using every worker.</p>
         */
        public FusedBuilder splitCount(int splitCount) {
            if (splitCount < 1) {
                throw new IllegalArgumentException("splitCount must be >= 1");
            }
            this.splitCount = splitCount;
            return this;
        }

        public Tensor tensor() {
            return new Tensor(new FusedNode(name, shape, execution,
                    Collections.unmodifiableMap(new LinkedHashMap<>(inputs)), outputInputName,
                    List.copyOf(refSteps), splitCount));
        }
    }

    private record FusedRefStep(String description, TensorOp op, String metricName, FusedRefMap map) {
        private FusedRefStep {
            Objects.requireNonNull(description, "description");
            Objects.requireNonNull(op, "op");
            Objects.requireNonNull(map, "map");
        }
    }

    private final class TimedNode implements Node {
        private final String metricName;
        private final Node delegate;

        private TimedNode(String metricName, Node delegate) {
            this.metricName = Objects.requireNonNull(metricName, "metricName");
            this.delegate = Objects.requireNonNull(delegate, "delegate");
        }

        @Override
        public RefEval evalRef() {
            Timer.Context timer = startTimer(metricName);
            try {
                return delegate.evalRef();
            } finally {
                stopTimer(timer);
            }
        }

        @Override
        public TensorShape shape() {
            return delegate.shape();
        }

        @Override
        public void render(StringBuilder sb, String indent, boolean last) {
            renderLine(sb, indent, last, "timer " + metricName + " -> " + delegate.label() + " " + compactShape(shape()));
            delegate.render(sb, indent + (last ? "   " : "│  "), true);
        }

        @Override
        public String label() {
            return delegate.label();
        }
    }

    private final class JustInTimeNode implements Node {
        private final Node delegate;
        private final int minSplit;
        private final int maxSplit;

        private JustInTimeNode(Node delegate, int minSplit, int maxSplit) {
            if (minSplit < 1 || maxSplit < minSplit) {
                throw new IllegalArgumentException("invalid justInTime split range");
            }
            this.delegate = Objects.requireNonNull(delegate, "delegate");
            this.minSplit = minSplit;
            this.maxSplit = maxSplit;
        }

        @Override
        public RefEval evalRef() {
            if (!(delegate instanceof FusedNode fused)) {
                throw new UnsupportedOperationException("justInTime currently supports fused TensorPlan nodes only");
            }
            if (adaptiveSplitTuner == null) {
                return fused.evalRef();
            }
            int max = Math.min(maxSplit, Math.toIntExact(Math.max(1, shape().size())));
            int min = Math.min(minSplit, max);
            int split = adaptiveSplitTuner.chooseSplit(fused.name, fused.shape(), fused.defaultSplitCount(), min, max);
            long start = System.nanoTime();
            try {
                return fused.evalRefWithSplitCount(split);
            } finally {
                adaptiveSplitTuner.observeSplit(fused.name, fused.shape(), split, System.nanoTime() - start);
            }
        }

        @Override
        public TensorShape shape() {
            return delegate.shape();
        }

        @Override
        public void render(StringBuilder sb, String indent, boolean last) {
            renderLine(sb, indent, last, "justInTime splits=" + minSplit + ".." + maxSplit + " -> "
                    + delegate.label() + " " + compactShape(shape()));
            delegate.render(sb, indent + (last ? "   " : "│  "), true);
        }

        @Override
        public String label() {
            return delegate.label();
        }
    }

    private interface Node {
        RefEval evalRef();

        TensorShape shape();

        void render(StringBuilder sb, String indent, boolean last);

        default String label() {
            return getClass().getSimpleName();
        }
    }

    private record RefInputNode(String name, TensorRef tensor, boolean mutable) implements Node {
        private RefInputNode {
            Objects.requireNonNull(name, "name");
            Objects.requireNonNull(tensor, "tensor");
        }

        @Override
        public RefEval evalRef() {
            return new RefEval(tensor, false, mutable);
        }

        @Override
        public TensorShape shape() {
            return tensor.shape();
        }

        @Override
        public void render(StringBuilder sb, String indent, boolean last) {
            renderLine(sb, indent, last, name + " " + compactShape(shape()) + " " + tensor.dType()
                    + " " + (mutable ? "mutable-ref" : "borrowed-ref"));
        }

        @Override
        public String label() {
            return name;
        }
    }

    private record RefImportedInputNode(String name, Node upstream, TensorRef tensor) implements Node {
        private RefImportedInputNode {
            Objects.requireNonNull(name, "name");
            Objects.requireNonNull(upstream, "upstream");
            Objects.requireNonNull(tensor, "tensor");
        }

        @Override
        public RefEval evalRef() {
            return new RefEval(tensor, false, false);
        }

        @Override
        public TensorShape shape() {
            return tensor.shape();
        }

        @Override
        public void render(StringBuilder sb, String indent, boolean last) {
            renderLine(sb, indent, last, name + " <- " + upstream.label() + " " + compactShape(shape())
                    + " " + tensor.dType() + " borrowed-ref");
        }

        @Override
        public String label() {
            return name;
        }
    }

    private record RefEval(TensorRef tensor, boolean owned, boolean mutable) {
    }

    private static void renderLine(StringBuilder sb, String indent, boolean last, String text) {
        sb.append(indent).append(last ? "└─ " : "├─ ").append(text).append('\n');
    }

    private record NamedNode(String name, Node delegate) implements Node {
        private NamedNode {
            Objects.requireNonNull(name, "name");
            Objects.requireNonNull(delegate, "delegate");
        }

        @Override
        public RefEval evalRef() {
            return delegate.evalRef();
        }

        @Override
        public TensorShape shape() {
            return delegate.shape();
        }

        @Override
        public void render(StringBuilder sb, String indent, boolean last) {
            renderLine(sb, indent, last, name + " = " + delegate.label() + " -> " + compactShape(shape()));
            delegate.render(sb, indent + (last ? "   " : "│  "), true);
        }

        @Override
        public String label() {
            return name;
        }
    }

    private final class BatchDotNode implements Node {
        private final Node input;
        private final Node weight;

        private BatchDotNode(Node input, Node weight) {
            this.input = input;
            this.weight = weight;
        }

        @Override
        public RefEval evalRef() {
            RefEval a = input.evalRef();
            RefEval b = weight.evalRef();
            TensorRef result = lighter.allocate(DType.F32, shape());
            try {
                lighter.batchDotProduct(new io.teknek.reckoning.tensor2.BatchDotProduct()
                        .result(result).a(a.tensor()).b(b.tensor())
                        .aColumnOffset(0).bColumnOffset(0).columnLength((int) a.tensor().shape().last())
                        .resultRowOffset(0).bRowOffset(0).rowChunkSize((int) b.tensor().shape().first()));
                closeRefIfOwned(a);
                closeRefIfOwned(b);
                return new RefEval(result, true, true);
            } catch (RuntimeException | Error e) {
                result.close();
                closeRefIfOwned(a);
                closeRefIfOwned(b);
                throw e;
            }
        }

        @Override
        public TensorShape shape() {
            return TensorShape.of((int) input.shape().first(), (int) weight.shape().first());
        }

        @Override
        public void render(StringBuilder sb, String indent, boolean last) {
            renderLine(sb, indent, last, "batchDot(" + input.label() + ", " + weight.label() + ") -> "
                    + compactShape(shape()));
            String childIndent = indent + (last ? "   " : "│  ");
            input.render(sb, childIndent, false);
            weight.render(sb, childIndent, true);
        }

        @Override
        public String label() {
            return "batchDot";
        }
    }

    private final class MlpNode implements Node {
        private final Node input;
        private final Node gateWeight;
        private final Node upWeight;
        private final Node downWeight;
        private final ActivationFunction.Type activation;
        private final DType quantizedType;

        private MlpNode(Node input, Node gateWeight, Node upWeight, Node downWeight,
                ActivationFunction.Type activation, DType quantizedType) {
            this.input = Objects.requireNonNull(input, "input");
            this.gateWeight = Objects.requireNonNull(gateWeight, "gateWeight");
            this.upWeight = Objects.requireNonNull(upWeight, "upWeight");
            this.downWeight = Objects.requireNonNull(downWeight, "downWeight");
            this.activation = Objects.requireNonNull(activation, "activation");
            this.quantizedType = Objects.requireNonNull(quantizedType, "quantizedType");
        }

        @Override
        public RefEval evalRef() {
            RefEval inputEval = input.evalRef();
            RefEval gateWeightEval = gateWeight.evalRef();
            RefEval upWeightEval = upWeight.evalRef();
            RefEval downWeightEval = downWeight.evalRef();
            int batchSize = (int) inputEval.tensor().shape().first();
            int embeddingLength = (int) inputEval.tensor().shape().last();
            int hiddenLength = (int) gateWeightEval.tensor().shape().first();
            int outputLength = (int) downWeightEval.tensor().shape().first();
            TensorRef gate = lighter.allocate(DType.F32, TensorShape.of(batchSize, hiddenLength));
            TensorRef up = lighter.allocate(DType.F32, TensorShape.of(batchSize, hiddenLength));
            TensorRef output = lighter.allocate(DType.F32, TensorShape.of(batchSize, outputLength));
            TensorRef downInput = null;
            try {
                projectBatch(gate, inputEval.tensor(), gateWeightEval.tensor(), embeddingLength, hiddenLength);
                projectBatch(up, inputEval.tensor(), upWeightEval.tensor(), embeddingLength, hiddenLength);
                for (int row = 0; row < batchSize; row++) {
                    for (int column = 0; column < hiddenLength; column++) {
                        gate.set(ActivationFunction.eval(activation, gate.get(row, column)), row, column);
                    }
                }
                lighter.multiplyAccumulate(new io.teknek.reckoning.tensor2.MultiplyAccumulate(up)
                        .into(gate).offsetAndLength(0, hiddenLength));
                downInput = gate.dType() == quantizedType ? gate : lighter.reshape(gate, quantizedType);
                projectBatch(output, downInput, downWeightEval.tensor(), hiddenLength, outputLength);
                closeRefIfOwned(inputEval);
                closeRefIfOwned(gateWeightEval);
                closeRefIfOwned(upWeightEval);
                closeRefIfOwned(downWeightEval);
                if (downInput != gate) {
                    downInput.close();
                }
                gate.close();
                up.close();
                return new RefEval(output, true, true);
            } catch (RuntimeException | Error e) {
                output.close();
                if (downInput != null && downInput != gate) {
                    downInput.close();
                }
                gate.close();
                up.close();
                closeRefIfOwned(inputEval);
                closeRefIfOwned(gateWeightEval);
                closeRefIfOwned(upWeightEval);
                closeRefIfOwned(downWeightEval);
                throw e;
            }
        }

        private void projectBatch(TensorRef output, TensorRef input, TensorRef weights,
                int columnLength, int rowCount) {
            lighter.batchDotProduct(new BatchDotProduct()
                    .result(output)
                    .a(input)
                    .b(weights)
                    .aColumnOffset(0)
                    .bColumnOffset(0)
                    .columnLength(columnLength)
                    .resultRowOffset(0)
                    .bRowOffset(0)
                    .rowChunkSize(rowCount)
                    .kernel(GemmKernel.BATCH)
                    .threading(Threading.PROVIDER));
        }

        @Override
        public TensorShape shape() {
            return TensorShape.of((int) input.shape().first(), (int) downWeight.shape().first());
        }

        @Override
        public void render(StringBuilder sb, String indent, boolean last) {
            renderLine(sb, indent, last, "mlp(" + input.label() + ", " + gateWeight.label() + ", "
                    + upWeight.label() + ", " + downWeight.label() + ") -> " + compactShape(shape())
                    + " activation=" + activation + " q=" + quantizedType);
            String childIndent = indent + (last ? "   " : "│  ");
            input.render(sb, childIndent, false);
            gateWeight.render(sb, childIndent, false);
            upWeight.render(sb, childIndent, false);
            downWeight.render(sb, childIndent, true);
        }

        @Override
        public String label() {
            return "mlp";
        }
    }

    private final class ActivationNode implements Node {
        private final Node input;
        private final ActivationFunction.Type activation;

        private ActivationNode(Node input, ActivationFunction.Type activation) {
            this.input = input;
            this.activation = activation;
        }

        @Override
        public RefEval evalRef() {
            RefEval in = input.evalRef();
            TensorRef output = lighter.allocate(in.tensor().dType(), in.tensor().shape());
            try {
                lighter.copy(in.tensor(), 0, output, 0, (int) in.tensor().shape().size());
                for (int row = 0; row < output.shape().first(); row++) {
                    for (int column = 0; column < output.shape().last(); column++) {
                        output.set(ActivationFunction.eval(activation, output.get(row, column)), row, column);
                    }
                }
                closeRefIfOwned(in);
                return new RefEval(output, true, true);
            } catch (RuntimeException | Error e) {
                output.close();
                closeRefIfOwned(in);
                throw e;
            }
        }

        @Override
        public TensorShape shape() {
            return input.shape();
        }

        @Override
        public void render(StringBuilder sb, String indent, boolean last) {
            renderLine(sb, indent, last, "activate " + activation + "(" + input.label() + ") -> "
                    + compactShape(shape()));
            input.render(sb, indent + (last ? "   " : "│  "), true);
        }

        @Override
        public String label() {
            return activation.name().toLowerCase();
        }
    }

    private final class MultiplyNode implements Node {
        private final Node left;
        private final Node right;

        private MultiplyNode(Node left, Node right) {
            this.left = left;
            this.right = right;
        }

        @Override
        public RefEval evalRef() {
            RefEval lhs = left.evalRef();
            RefEval rhs = right.evalRef();
            Preconditions.checkArgument(lhs.tensor().dims() == 2 && rhs.tensor().dims() == 2,
                    "TensorRef multiply currently requires 2D tensors");
            TensorRef output = lighter.allocate(lhs.tensor().dType(), lhs.tensor().shape());
            try {
                lighter.copy(lhs.tensor(), 0, output, 0, (int) lhs.tensor().shape().size());
                lighter.multiplyAccumulate(new io.teknek.reckoning.tensor2.MultiplyAccumulate(rhs.tensor())
                        .into(output).offsetAndLength(0, (int) output.shape().last()));
                closeRefIfOwned(lhs);
                closeRefIfOwned(rhs);
                return new RefEval(output, true, true);
            } catch (RuntimeException | Error e) {
                output.close();
                closeRefIfOwned(lhs);
                closeRefIfOwned(rhs);
                throw e;
            }
        }

        @Override
        public TensorShape shape() {
            return left.shape();
        }

        @Override
        public void render(StringBuilder sb, String indent, boolean last) {
            String text = left instanceof ActivationNode activationNode
                    ? "fused " + activationNode.activation.name().toLowerCase() + "("
                        + activationNode.input.label() + ") * " + right.label()
                    : "multiply(" + left.label() + ", " + right.label() + ")";
            renderLine(sb, indent, last, text + " -> " + compactShape(shape()));
            String childIndent = indent + (last ? "   " : "│  ");
            left.render(sb, childIndent, false);
            right.render(sb, childIndent, true);
        }

        @Override
        public String label() {
            return "multiply";
        }
    }

    private final class AddNode implements Node {
        private final Node left;
        private final Node right;

        private AddNode(Node left, Node right) {
            this.left = left;
            this.right = right;
        }

        @Override
        public RefEval evalRef() {
            RefEval lhs = left.evalRef();
            RefEval rhs = right.evalRef();
            Preconditions.checkArgument(lhs.tensor().dims() == 2 && rhs.tensor().dims() == 2,
                    "TensorRef add currently requires 2D tensors");
            Preconditions.checkArgument(lhs.tensor().shape().equals(rhs.tensor().shape()),
                    "TensorRef add inputs must have matching shapes");
            TensorRef output = lighter.allocate(lhs.tensor().dType(), lhs.tensor().shape());
            try {
                lighter.copy(lhs.tensor(), 0, output, 0, (int) lhs.tensor().shape().size());
                lighter.accumulate(new io.teknek.reckoning.tensor2.Accumulate(rhs.tensor())
                        .into(output).offsetAndLength(0, (int) output.shape().last()));
                closeRefIfOwned(lhs);
                closeRefIfOwned(rhs);
                return new RefEval(output, true, true);
            } catch (RuntimeException | Error e) {
                output.close();
                closeRefIfOwned(lhs);
                closeRefIfOwned(rhs);
                throw e;
            }
        }

        @Override
        public TensorShape shape() {
            return left.shape();
        }

        @Override
        public void render(StringBuilder sb, String indent, boolean last) {
            renderLine(sb, indent, last, "add(" + left.label() + ", " + right.label() + ") -> " + compactShape(shape()));
            String childIndent = indent + (last ? "   " : "│  ");
            left.render(sb, childIndent, false);
            right.render(sb, childIndent, true);
        }

        @Override
        public String label() {
            return "add";
        }
    }

    private final class ScaleNode implements Node {
        private final Node input;
        private final float factor;

        private ScaleNode(Node input, float factor) {
            this.input = input;
            this.factor = factor;
        }

        @Override
        public RefEval evalRef() {
            RefEval in = input.evalRef();
            TensorRef output = lighter.allocate(in.tensor().dType(), in.tensor().shape());
            try {
                lighter.copy(in.tensor(), 0, output, 0, (int) in.tensor().shape().size());
                lighter.scale(new Scale(factor).target(output).offsetAndLength(0,
                        (int) output.shape().last()));
                closeRefIfOwned(in);
                return new RefEval(output, true, true);
            } catch (RuntimeException | Error e) {
                output.close();
                closeRefIfOwned(in);
                throw e;
            }
        }

        @Override
        public TensorShape shape() {
            return input.shape();
        }

        @Override
        public void render(StringBuilder sb, String indent, boolean last) {
            renderLine(sb, indent, last, "scale(" + factor + ", " + input.label() + ") -> " + compactShape(shape()));
            input.render(sb, indent + (last ? "   " : "│  "), true);
        }

        @Override
        public String label() {
            return "scale";
        }
    }

    private final class QuantizeNode implements Node {
        private final Node input;
        private final DType dtype;

        private QuantizeNode(Node input, DType dtype) {
            this.input = input;
            this.dtype = dtype;
        }

        @Override
        public RefEval evalRef() {
            RefEval in = input.evalRef();
            try {
                TensorRef output = lighter.reshape(in.tensor(), dtype);
                closeRefIfOwned(in);
                return new RefEval(output, true, true);
            } catch (RuntimeException | Error e) {
                closeRefIfOwned(in);
                throw e;
            }
        }

        @Override
        public TensorShape shape() {
            return input.shape();
        }

        @Override
        public void render(StringBuilder sb, String indent, boolean last) {
            renderLine(sb, indent, last, "quantize(" + dtype + ", " + input.label() + ") -> " + compactShape(shape()));
            input.render(sb, indent + (last ? "   " : "│  "), true);
        }

        @Override
        public String label() {
            return "quantize";
        }
    }

    private final class FusedNode implements Node {
        private final String name;
        private final TensorShape shape;
        private final FusedExecution execution;
        private final Map<String, Node> inputs;
        private final String outputInputName;
        private final List<FusedRefStep> refSteps;
        private final Integer splitCount;

        private FusedNode(String name, TensorShape shape, FusedExecution execution, Map<String, Node> inputs,
                String outputInputName, List<FusedRefStep> refSteps, Integer splitCount) {
            this.name = name;
            this.shape = shape;
            this.execution = execution;
            this.inputs = inputs;
            this.outputInputName = outputInputName;
            this.refSteps = refSteps;
            this.splitCount = splitCount;
        }

        @Override
        public RefEval evalRef() {
            return evalRefWithSplitCount(splitCount);
        }

        private RefEval evalRefWithSplitCount(Integer splitOverride) {
            if (refSteps.isEmpty()) {
                throw new UnsupportedOperationException("TensorRef fused node has no Ref callbacks: " + name);
            }
            Map<String, RefEval> evals = new LinkedHashMap<>();
            Map<String, TensorRef> tensors = new LinkedHashMap<>();
            for (Map.Entry<String, Node> entry : inputs.entrySet()) {
                RefEval eval = entry.getValue().evalRef();
                evals.put(entry.getKey(), eval);
                tensors.put(entry.getKey(), eval.tensor());
            }
            RefEval outputEval = outputInputName == null ? null : evals.get(outputInputName);
            TensorRef output = outputEval == null ? lighter.allocate(DType.F32, shape) : outputEval.tensor();
            tensors.put(name, output);
            executeRef(tensors, splitOverride);
            evals.entrySet().stream()
                    .filter(entry -> outputInputName == null || !outputInputName.equals(entry.getKey()))
                    .map(Map.Entry::getValue)
                    .forEach(TensorPlan::closeRefIfOwned);
            return new RefEval(output, outputEval == null || outputEval.owned(), true);
        }

        private void executeRef(Map<String, TensorRef> tensors, Integer splitOverride) {
            FusedRefContext context = new FusedRefContext(tensors);
            if (execution == FusedExecution.INT_STREAM_COLUMNS) {
                IntStream.range(0, (int) shape.last()).parallel()
                        .forEach(column -> runRefSteps(context, column, 1));
            } else if (execution == FusedExecution.INT_STREAM_ROWS) {
                IntStream.range(0, (int) shape.first()).parallel()
                        .forEach(row -> runRefSteps(context, row, 1));
            } else {
                List<TensorSplit> splits = TensorLib.calculateTSplits(0, shape.size(),
                        splitOverride == null ? defaultSplitCount() : splitOverride);
                List<ForkJoinTask<?>> tasks = new ArrayList<>();
                for (TensorSplit split : splits) {
                    tasks.add(pool.getUnderlying().submit(() -> runRefSteps(context, split.offset, split.length)));
                }
                tasks.forEach(ForkJoinTask::join);
            }
        }

        private void runRefSteps(FusedRefContext context, long offset, long length) {
            for (FusedRefStep step : refSteps) {
                Timer.Context timer = startTimer(step.metricName());
                try {
                    step.map().map(context, offset, length);
                } finally {
                    stopTimer(timer);
                }
            }
        }

        private int defaultSplitCount() {
            return splitCount == null ? Math.max(1, pool.getCoreCount()) : splitCount;
        }

        @Override
        public TensorShape shape() {
            return shape;
        }

        @Override
        public void render(StringBuilder sb, String indent, boolean last) {
            renderLine(sb, indent, last, name + " = fuse -> " + compactShape(shape));
            String childIndent = indent + (last ? "   " : "│  ");
            int idx = 0;
            for (FusedRefStep step : refSteps) {
                renderLine(sb, childIndent, false, "mapRef " + (idx++) + " [" + step.op() + "] " + step.description());
            }
            int inputIndex = 0;
            for (Map.Entry<String, Node> entry : inputs.entrySet()) {
                entry.getValue().render(sb, childIndent, ++inputIndex == inputs.size());
            }
        }

        @Override
        public String label() {
            return name;
        }
    }

    private static String compactShape(TensorShape shape) {
        int[] raw = shape.shapeArray();
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < raw.length; i++) {
            if (i > 0) {
                sb.append('x');
            }
            sb.append(raw[i]);
        }
        sb.append(']');
        return sb.toString();
    }

    private Timer.Context startTimer(String metricName) {
        return metricRegistry == null || metricName == null ? null : metricRegistry.timer(metricName).time();
    }

    private static void stopTimer(Timer.Context timer) {
        if (timer != null) {
            timer.stop();
        }
    }

    private static void closeRefIfOwned(RefEval eval) {
        if (eval.owned()) {
            eval.tensor().close();
        }
    }

    private boolean useTensorRuntime() {
        return runtime != null && runMode != RunMode.CALLER_THREAD;
    }
}
