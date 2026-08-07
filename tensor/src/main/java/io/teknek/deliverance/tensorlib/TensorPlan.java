package io.teknek.deliverance.tensorlib;

import com.codahale.metrics.MetricRegistry;
import com.codahale.metrics.Timer;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.operations.TensorOperations;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Small lazy tensor workflow for experimenting with logical tensor plans and local fusion rules.
 *
 * <p>External tensors are introduced with {@link #input(AbstractTensor)} or {@link #mutable(AbstractTensor)}. Operations
 * build a logical graph and execute only when {@link Tensor#materialize()} is called.</p>
 */
public final class TensorPlan {
    private static final Logger LOGGER = LoggerFactory.getLogger(TensorPlan.class);
    public enum RunMode {
        DEFAULT,
        CALLER_THREAD
    }

    private final TensorOperations operations;
    private final WrappedForkJoinPool pool;
    private final MetricRegistry metricRegistry;
    private final TensorRuntime runtime;
    private RunMode runMode = RunMode.DEFAULT;

    public TensorPlan(TensorOperations operations, WrappedForkJoinPool pool) {
        this(operations, pool, null);
    }

    public TensorPlan(TensorOperations operations, WrappedForkJoinPool pool, MetricRegistry metricRegistry) {
        this(operations, pool, metricRegistry, null);
    }

    public TensorPlan(TensorOperations operations, WrappedForkJoinPool pool, MetricRegistry metricRegistry,
            TensorRuntime runtime) {
        this.operations = Objects.requireNonNull(operations, "operations");
        this.pool = Objects.requireNonNull(pool, "pool");
        this.metricRegistry = metricRegistry;
        this.runtime = runtime;
    }

    public Tensor input(AbstractTensor tensor) {
        return input("input", tensor);
    }

    public Tensor input(String name, AbstractTensor tensor) {
        ensureLocality(tensor);
        return new Tensor(new InputNode(name, tensor, false));
    }

    public ImmutableTensor immutable(String name, AbstractTensor tensor) {
        ensureLocality(tensor);
        return new ImmutableTensor(new InputNode(name, tensor, false));
    }

    public Tensor mutable(AbstractTensor tensor) {
        return mutable("mutable", tensor);
    }

    public Tensor mutable(String name, AbstractTensor tensor) {
        ensureLocality(tensor);
        return new Tensor(new InputNode(name, tensor, true));
    }

    public FusedBuilder fuse(String name, TensorShape shape) {
        return new FusedBuilder(name, shape, FusedExecution.FIXED_POOL_LINEAR);
    }

    public FusedBuilder fuseColumnsIntStream(String name, TensorShape shape) {
        return new FusedBuilder(name, shape, FusedExecution.INT_STREAM_COLUMNS);
    }

    public FusedBuilder fuseRowsIntStream(String name, TensorShape shape) {
        return new FusedBuilder(name, shape, FusedExecution.INT_STREAM_ROWS);
    }

    public TensorPlan forcedRunMode(RunMode runMode) {
        this.runMode = Objects.requireNonNull(runMode, "runMode");
        return this;
    }

    /** Computes a dot product over two row slices. */
    public float dotSlice(AbstractTensor left, int leftRow, int leftOffset, AbstractTensor right, int rightRow,
            int rightOffset, int length) {
        Objects.requireNonNull(left, "left");
        Objects.requireNonNull(right, "right");
        recordPrimitivePath("tensorplan.dot_slice", operations.usesOptimizedDotSlice(left, right));
        float[] result = new float[1];
        run("tensorplan.dot_slice", 0, Optional.of(left), () -> result[0] = operations.dotSlice(left, leftRow, leftOffset,
                right, rightRow, rightOffset, length));
        return result[0];
    }

    public void dotRowsToArray(AbstractTensor left, int leftRow, int leftOffset, AbstractTensor rows, int rowOffset,
            int rowColumnOffset, int rowCount, int width, float[] scores, int scoresOffset) {
        Objects.requireNonNull(left, "left");
        Objects.requireNonNull(rows, "rows");
        Objects.requireNonNull(scores, "scores");
        recordPrimitivePath("tensorplan.dot_rows_to_array", operations.usesOptimizedDotRowsToArray(left, rows));
        run("tensorplan.dot_rows_to_array", 0, Optional.of(left), () -> operations.dotRowsToArray(left, leftRow,
                leftOffset, rows, rowOffset, rowColumnOffset, rowCount, width, scores, scoresOffset));
    }

    /** Mutates {@code out = out * oldScale + value * weight} over one row slice. */
    public void weightedRescaleAccumulateSlice(AbstractTensor out, int outRow, int outOffset, AbstractTensor value,
            int valueRow, int valueOffset, int length, float oldScale, float weight) {
        Objects.requireNonNull(out, "out");
        Objects.requireNonNull(value, "value");
        recordPrimitivePath("tensorplan.weighted_rescale_accumulate_slice",
                operations.usesOptimizedWeightedRescaleAccumulateSlice(out, value));
        run("tensorplan.weighted_rescale_accumulate_slice", 0, Optional.of(out), () ->
                operations.weightedRescaleAccumulateSlice(out, outRow, outOffset, value, valueRow, valueOffset, length,
                        oldScale, weight));
    }

    /** Mutates {@code out += value * weight} over one row slice. */
    public void accumulateWeightedSlice(AbstractTensor out, int outRow, int outOffset, AbstractTensor value,
            int valueRow, int valueOffset, int length, float weight) {
        Objects.requireNonNull(out, "out");
        Objects.requireNonNull(value, "value");
        recordPrimitivePath("tensorplan.accumulate_weighted_slice",
                operations.usesOptimizedAccumulateWeightedSlice(out, value));
        run("tensorplan.accumulate_weighted_slice", 0, Optional.of(out), () ->
                operations.accumulateWeightedSlice(out, outRow, outOffset, value, valueRow, valueOffset, length,
                        weight));
    }

    public void accumulateWeightedRows(AbstractTensor out, int outRow, int outOffset, AbstractTensor rows,
            int rowOffset, int rowColumnOffset, int rowCount, int width, float[] weights, int weightsOffset) {
        Objects.requireNonNull(out, "out");
        Objects.requireNonNull(rows, "rows");
        Objects.requireNonNull(weights, "weights");
        recordPrimitivePath("tensorplan.accumulate_weighted_rows",
                operations.usesOptimizedAccumulateWeightedRows(out, rows));
        run("tensorplan.accumulate_weighted_rows", 0, Optional.of(out), () -> operations.accumulateWeightedRows(out,
                outRow, outOffset, rows, rowOffset, rowColumnOffset, rowCount, width, weights, weightsOffset));
    }

    /** Multiplies one row slice by {@code factor}. */
    public void normalizeSlice(AbstractTensor tensor, int row, int offset, int length, float factor) {
        Objects.requireNonNull(tensor, "tensor");
        recordPrimitivePath("tensorplan.normalize_slice", operations.usesOptimizedNormalizeSlice(tensor));
        run("tensorplan.normalize_slice", 0, Optional.of(tensor), () -> operations.normalizeSlice(tensor, row, offset,
                length, factor));
    }

    public void scaleSlice(AbstractTensor tensor, int row, int offset, int length, float factor) {
        Objects.requireNonNull(tensor, "tensor");
        recordPrimitivePath("tensorplan.scale_slice", operations.usesOptimizedScaleSlice(tensor));
        run("tensorplan.scale_slice", 0, Optional.of(tensor), () -> operations.scaleSlice(tensor, row, offset,
                length, factor));
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

        public TensorShape shape() {
            return node.shape();
        }

        public AbstractTensor materialize() {
            if (LOGGER.isDebugEnabled()) {
                LOGGER.debug("TensorPlan:\n{}", plan());
            }
            return node.eval().tensor();
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
    public interface FusedMap {
        void map(FusedContext context, long offset, long length);
    }

    public final class FusedContext {
        private final Map<String, AbstractTensor> tensors;

        private FusedContext(Map<String, AbstractTensor> tensors) {
            this.tensors = tensors;
        }

        public AbstractTensor tensor(String name) {
            AbstractTensor tensor = tensors.get(name);
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
        private final List<FusedStep> steps = new ArrayList<>();
        private String outputInputName;

        private FusedBuilder(String name, TensorShape shape, FusedExecution execution) {
            this.name = Objects.requireNonNull(name, "name");
            this.shape = Objects.requireNonNull(shape, "shape");
            this.execution = Objects.requireNonNull(execution, "execution");
        }

        public FusedBuilder read(String name, Tensor tensor) {
            inputs.put(name, tensor.node);
            return this;
        }

        public FusedBuilder write(String name, Tensor tensor) {
            read(name, tensor);
            outputInputName = name;
            return this;
        }

        public FusedBuilder map(String description, TensorOp op, FusedMap map) {
            return map(description, op, null, map);
        }

        public FusedBuilder map(String description, TensorOp op, String metricName, FusedMap map) {
            steps.add(new FusedStep(description, op, metricName, map));
            return this;
        }

        public Tensor tensor() {
            return new Tensor(new FusedNode(name, shape, execution, Map.copyOf(inputs), outputInputName,
                    List.copyOf(steps)));
        }
    }

    private record FusedStep(String description, TensorOp op, String metricName, FusedMap map) {
        private FusedStep {
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
        public Eval eval() {
            Timer.Context timer = startTimer(metricName);
            try {
                return delegate.eval();
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

    private interface Node {
        Eval eval();

        TensorShape shape();

        void render(StringBuilder sb, String indent, boolean last);

        default String label() {
            return getClass().getSimpleName();
        }
    }

    private record Eval(AbstractTensor tensor, boolean owned, boolean mutable) {
    }

    private static void renderLine(StringBuilder sb, String indent, boolean last, String text) {
        sb.append(indent).append(last ? "└─ " : "├─ ").append(text).append('\n');
    }

    private record InputNode(String name, AbstractTensor tensor, boolean mutable) implements Node {
        private InputNode {
            Objects.requireNonNull(name, "name");
            Objects.requireNonNull(tensor, "tensor");
        }

        @Override
        public Eval eval() {
            return new Eval(tensor, false, mutable);
        }

        @Override
        public TensorShape shape() {
            return tensor.shape();
        }

        @Override
        public void render(StringBuilder sb, String indent, boolean last) {
            renderLine(sb, indent, last, name + " " + compactShape(shape()) + " " + tensor.dType()
                    + " " + (mutable ? "mutable" : "borrowed"));
        }

        @Override
        public String label() {
            return name;
        }
    }

    private record NamedNode(String name, Node delegate) implements Node {
        private NamedNode {
            Objects.requireNonNull(name, "name");
            Objects.requireNonNull(delegate, "delegate");
        }

        @Override
        public Eval eval() {
            return delegate.eval();
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
        public Eval eval() {
            Eval a = input.eval();
            Eval b = weight.eval();
            AbstractTensor result = new FloatBufferTensor((int) a.tensor().shape().first(), (int) b.tensor().shape().first());
            a.tensor().locality().ifPresent(result::setLocality);
            run("tensorplan.batchDot", 0, Optional.of(a.tensor()), () -> operations.dotProductChunk(result, a.tensor(),
                    b.tensor(), 0, (int) a.tensor().shape().last(), 0, (int) b.tensor().shape().first()));
            closeIfOwned(a);
            closeIfOwned(b);
            return new Eval(result, true, true);
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
        public Eval eval() {
            Eval inputEval = input.eval();
            Eval gateWeightEval = gateWeight.eval();
            Eval upWeightEval = upWeight.eval();
            Eval downWeightEval = downWeight.eval();
            int batchSize = (int) inputEval.tensor().shape().first();
            int embeddingLength = (int) inputEval.tensor().shape().last();
            int hiddenLength = (int) gateWeightEval.tensor().shape().first();
            int outputLength = (int) downWeightEval.tensor().shape().first();
            AbstractTensor gate = new FloatBufferTensor(batchSize, hiddenLength);
            AbstractTensor up = new FloatBufferTensor(batchSize, hiddenLength);
            AbstractTensor[] hidden = new AbstractTensor[1];
            AbstractTensor output = new FloatBufferTensor(batchSize, outputLength);
            inputEval.tensor().locality().ifPresent(output::setLocality);
            try {
                Runnable compute = () -> {
                    AbstractTensor[] projectionResults = new AbstractTensor[] { gate, up };
                    AbstractTensor[] projectionWeights = new AbstractTensor[] { gateWeightEval.tensor(), upWeightEval.tensor() };
                    Timer.Context gateUpTimer = startTimer("tensorplan.mlp.gate_up_projection");
                    try {
                        runProviderRowChunks(hiddenLength,
                                (chunkStart, chunkSize) -> operations.dotProductBatchChunk(projectionResults,
                                        inputEval.tensor(), projectionWeights, 0, embeddingLength, chunkStart, chunkSize));
                    } finally {
                        stopTimer(gateUpTimer);
                    }
                    Timer.Context activationTimer = startTimer("tensorplan.mlp.fused_activation_multiply_quantize");
                    try {
                        hidden[0] = operations.activationMultiplyQuantize(gate, up, activation, quantizedType, 0,
                                hiddenLength);
                    } finally {
                        stopTimer(activationTimer);
                    }
                    hidden[0].locality().or(() -> inputEval.tensor().locality()).ifPresent(output::setLocality);
                    AbstractTensor hiddenTensor = hidden[0];
                    Timer.Context downTimer = startTimer("tensorplan.mlp.down_projection");
                    try {
                        runProviderRowChunks(outputLength,
                                (chunkStart, chunkSize) -> operations.dotProductChunk(output, hiddenTensor,
                                        downWeightEval.tensor(), 0, hiddenLength, chunkStart, chunkSize));
                    } finally {
                        stopTimer(downTimer);
                    }
                };
                if (useTensorRuntime()) {
                    runtime.runAndWait("tensorplan.mlp", 0, Optional.of(inputEval.tensor()), compute);
                } else {
                    compute.run();
                }
                return new Eval(output, true, true);
            } finally {
                gate.close();
                up.close();
                if (hidden[0] != null) {
                    hidden[0].close();
                }
                closeIfOwned(inputEval);
                closeIfOwned(gateWeightEval);
                closeIfOwned(upWeightEval);
                closeIfOwned(downWeightEval);
            }
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
        public Eval eval() {
            Eval in = input.eval();
            AbstractTensor out = in.owned() || in.mutable() ? in.tensor() : copyOf(in.tensor());
            // Optimization gap: TensorOperations has no activation primitive today, so activation remains a TensorPlan
            // Java loop. Add a provider-backed activation op or fused physical lowering before relying on this broadly.
            applyActivationInPlace(out, activation);
            return new Eval(out, in.owned(), in.mutable());
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
        public Eval eval() {
            if (left instanceof ActivationNode activationNode) {
                Eval base = activationNode.input.eval();
                Eval rhs = right.eval();
                AbstractTensor out = base.owned() || base.mutable() ? base.tensor() : copyOf(base.tensor());
                // Optimization gap: this fused activation*multiply avoids an extra pass, but still performs activation
                // through a Java loop because TensorOperations has no provider-backed activation primitive yet.
                applyActivationMultiplyInPlace(out, rhs.tensor(), activationNode.activation);
                closeIfOwned(rhs);
                return new Eval(out, base.owned(), base.mutable());
            }
            Eval lhs = left.eval();
            Eval rhs = right.eval();
            AbstractTensor out = lhs.owned() || lhs.mutable() ? lhs.tensor() : copyOf(lhs.tensor());
            run("tensorplan.multiply", 0, Optional.of(lhs.tensor()), () -> operations.maccumulate(out, rhs.tensor(), 0,
                    (int) out.shape().last()));
            closeIfOwned(rhs);
            return new Eval(out, lhs.owned(), lhs.mutable());
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
        public Eval eval() {
            Eval lhs = left.eval();
            Eval rhs = right.eval();
            AbstractTensor out = lhs.owned() || lhs.mutable() ? lhs.tensor() : copyOf(lhs.tensor());
            run("tensorplan.add", 0, Optional.of(out), () -> operations.accumulate(out, rhs.tensor(), 0,
                    (int) out.shape().last()));
            closeIfOwned(rhs);
            return new Eval(out, lhs.owned(), lhs.mutable());
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
        public Eval eval() {
            Eval in = input.eval();
            AbstractTensor out = in.owned() || in.mutable() ? in.tensor() : copyOf(in.tensor());
            run("tensorplan.scale", 0, Optional.of(out), () -> operations.scale(factor, out, 0,
                    (int) out.shape().last()));
            return new Eval(out, in.owned(), in.mutable());
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
        public Eval eval() {
            if (input instanceof MultiplyNode multiplyNode && multiplyNode.left instanceof ActivationNode activationNode) {
                Eval base = activationNode.input.eval();
                Eval rhs = multiplyNode.right.eval();
                final AbstractTensor[] result = new AbstractTensor[1];
                run("tensorplan.activation_multiply_quantize", 0, Optional.of(base.tensor()),
                        () -> result[0] = operations.activationMultiplyQuantize(base.tensor(), rhs.tensor(),
                                activationNode.activation, dtype, 0, (int) base.tensor().shape().last()));
                base.tensor().locality().ifPresent(result[0]::setLocality);
                closeIfOwned(base);
                closeIfOwned(rhs);
                return new Eval(result[0], true, true);
            }
            Eval in = input.eval();
            final AbstractTensor[] result = new AbstractTensor[1];
            run("tensorplan.quantize", 0, Optional.of(in.tensor()), () -> result[0] = operations.quantize(in.tensor(),
                    dtype, 0, (int) in.tensor().shape().last()));
            AbstractTensor out = result[0];
            in.tensor().locality().ifPresent(out::setLocality);
            closeIfOwned(in);
            return new Eval(out, true, true);
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
        private final List<FusedStep> steps;

        private FusedNode(String name, TensorShape shape, FusedExecution execution, Map<String, Node> inputs,
                String outputInputName, List<FusedStep> steps) {
            this.name = name;
            this.shape = shape;
            this.execution = execution;
            this.inputs = inputs;
            this.outputInputName = outputInputName;
            this.steps = steps;
        }

        @Override
        public Eval eval() {
            Map<String, Eval> evals = new LinkedHashMap<>();
            Map<String, AbstractTensor> tensors = new LinkedHashMap<>();
            for (Map.Entry<String, Node> entry : inputs.entrySet()) {
                Eval eval = entry.getValue().eval();
                evals.put(entry.getKey(), eval);
                tensors.put(entry.getKey(), eval.tensor());
            }
            Eval outputEval = outputInputName == null ? null : evals.get(outputInputName);
            AbstractTensor output = outputEval == null ? new FloatBufferTensor(shape) : outputEval.tensor();
            if (outputEval == null) {
                tensors.values().stream().findFirst().flatMap(AbstractTensor::locality).ifPresent(output::setLocality);
            }
            tensors.put(name, output);
            FusedContext context = new FusedContext(tensors);
            if (execution == FusedExecution.INT_STREAM_COLUMNS) {
                executeIntStreamColumns(context);
            } else if (execution == FusedExecution.INT_STREAM_ROWS) {
                executeIntStreamRows(context);
            } else {
                executeFixedPoolLinear(context);
            }
            evals.entrySet().stream()
                    .filter(entry -> outputInputName == null || !outputInputName.equals(entry.getKey()))
                    .map(Map.Entry::getValue)
                    .forEach(TensorPlan::closeIfOwned);
            return new Eval(output, outputEval == null || outputEval.owned(), true);
        }

        private void executeFixedPoolLinear(FusedContext context) {
            long length = shape.size();
            List<TensorSplit> splits = TensorLib.calculateTSplits(0, length, Math.max(1, pool.getCoreCount()));
            if (useTensorRuntime()) {
                List<CompletableFuture<Void>> tasks = new ArrayList<>();
                int chunk = 0;
                for (TensorSplit split : splits) {
                    tasks.add(runtime.submit("tensorplan.fuse.linear", chunk++, representativeTensor(context),
                            () -> runSteps(context, split.offset, split.length)));
                }
                tasks.forEach(CompletableFuture::join);
                return;
            }
            List<ForkJoinTask<?>> tasks = new ArrayList<>();
            for (TensorSplit split : splits) {
                tasks.add(pool.getUnderlying().submit(() -> runSteps(context, split.offset, split.length)));
            }
            tasks.forEach(ForkJoinTask::join);
        }

        private void executeIntStreamColumns(FusedContext context) {
            int columns = (int) shape.last();
            if (useTensorRuntime()) {
                List<CompletableFuture<Void>> tasks = new ArrayList<>();
                for (int column = 0; column < columns; column++) {
                    int chunk = column;
                    tasks.add(runtime.submit("tensorplan.fuse.columns", chunk, representativeTensor(context),
                            () -> runSteps(context, chunk, 1)));
                }
                tasks.forEach(CompletableFuture::join);
                return;
            }
            IntStream.range(0, columns).parallel().forEach(column -> runSteps(context, column, 1));
        }

        private void executeIntStreamRows(FusedContext context) {
            int rows = (int) shape.first();
            if (useTensorRuntime()) {
                List<CompletableFuture<Void>> tasks = new ArrayList<>();
                for (int row = 0; row < rows; row++) {
                    int chunk = row;
                    tasks.add(runtime.submit("tensorplan.fuse.rows", chunk, representativeTensor(context),
                            () -> runSteps(context, chunk, 1)));
                }
                tasks.forEach(CompletableFuture::join);
                return;
            }
            IntStream.range(0, rows).parallel().forEach(row -> runSteps(context, row, 1));
        }

        private Optional<AbstractTensor> representativeTensor(FusedContext context) {
            if (outputInputName != null) {
                return Optional.ofNullable(context.tensors.get(outputInputName));
            }
            return context.tensors.values().stream().findFirst();
        }

        private void runSteps(FusedContext context, long offset, long length) {
            for (FusedStep step : steps) {
                Timer.Context timer = startTimer(step.metricName());
                try {
                    step.map().map(context, offset, length);
                } finally {
                    stopTimer(timer);
                }
            }
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
            for (FusedStep step : steps) {
                renderLine(sb, childIndent, false, "map " + (idx++) + " [" + step.op() + "] " + step.description());
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

    private void recordPrimitivePath(String operation, boolean optimized) {
        if (metricRegistry == null) {
            return;
        }
        metricRegistry.counter(operation + (optimized ? ".optimized" : ".generic")).inc();
    }

    private static void closeIfOwned(Eval eval) {
        if (eval.owned()) {
            eval.tensor().close();
        }
    }

    private static AbstractTensor copyOf(AbstractTensor tensor) {
        FloatBufferTensor copy = new FloatBufferTensor(tensor.shape());
        copy.copyFrom(tensor, 0, 0, (int) tensor.size());
        tensor.locality().ifPresent(copy::setLocality);
        return copy;
    }

    private void applyActivationInPlace(AbstractTensor tensor, ActivationFunction.Type activation) {
        forEachElement(tensor, (row, col) -> tensor.set(ActivationFunction.eval(activation, tensor.get(row, col)), row, col));
    }

    private void applyActivationMultiplyInPlace(AbstractTensor lhs, AbstractTensor rhs,
            ActivationFunction.Type activation) {
        forEachElement(lhs, (row, col) -> lhs.set(ActivationFunction.eval(activation, lhs.get(row, col))
                * rhs.get(row, col), row, col));
    }

    private void multiplyInPlace(AbstractTensor lhs, AbstractTensor rhs) {
        forEachElement(lhs, (row, col) -> lhs.set(lhs.get(row, col) * rhs.get(row, col), row, col));
    }

    private void forEachElement(AbstractTensor tensor, ElementConsumer consumer) {
        int rows = (int) tensor.shape().first();
        int cols = (int) tensor.shape().last();
        long length = (long) rows * cols;
        List<TensorSplit> splits = TensorLib.calculateTSplits(0, length, Math.max(1, pool.getCoreCount()));
        if (useTensorRuntime()) {
            List<CompletableFuture<Void>> tasks = new ArrayList<>();
            int chunk = 0;
            for (TensorSplit split : splits) {
                tasks.add(runtime.submit("tensorplan.foreach", chunk++, Optional.of(tensor), () -> {
                    long end = split.offset + split.length;
                    for (long index = split.offset; index < end; index++) {
                        int row = (int) (index / cols);
                        int col = (int) (index % cols);
                        consumer.accept(row, col);
                    }
                }));
            }
            tasks.forEach(CompletableFuture::join);
            return;
        }
        List<ForkJoinTask<?>> tasks = new ArrayList<>();
        for (TensorSplit split : splits) {
            tasks.add(pool.getUnderlying().submit(() -> {
                long end = split.offset + split.length;
                for (long index = split.offset; index < end; index++) {
                    int row = (int) (index / cols);
                    int col = (int) (index % cols);
                    consumer.accept(row, col);
                }
            }));
        }
        tasks.forEach(ForkJoinTask::join);
    }

    @FunctionalInterface
    private interface ElementConsumer {
        void accept(int row, int col);
    }

    private void run(String operation, int chunkId, Optional<AbstractTensor> tensor, Runnable action) {
        if (runtime == null) {
            action.run();
            return;
        }
        if (!useTensorRuntime()) {
            action.run();
            return;
        }
        runtime.runAndWait(operation, chunkId, tensor, action);
    }

    private void runRowChunks(String operation, int rowCount, AbstractTensor representative, RowChunk action) {
        List<TensorSplit> splits = TensorLib.calculateTSplits(0, rowCount, Math.max(1, operations.parallelSplitSize()));
        if (useTensorRuntime()) {
            List<CompletableFuture<Void>> tasks = new ArrayList<>();
            int chunk = 0;
            for (TensorSplit split : splits) {
                int chunkStart = (int) split.offset;
                int chunkSize = (int) split.length;
                tasks.add(runtime.submit(operation, chunk++, Optional.of(representative),
                        () -> action.run(chunkStart, chunkSize)));
            }
            tasks.forEach(CompletableFuture::join);
            return;
        }
        List<ForkJoinTask<?>> tasks = new ArrayList<>();
        for (TensorSplit split : splits) {
            int chunkStart = (int) split.offset;
            int chunkSize = (int) split.length;
            tasks.add(pool.getUnderlying().submit(() -> action.run(chunkStart, chunkSize)));
        }
        tasks.forEach(ForkJoinTask::join);
    }

    private void runProviderRowChunks(int rowCount, RowChunk action) {
        List<TensorSplit> splits = TensorLib.calculateTSplits(0, rowCount, Math.max(1, operations.parallelSplitSize()));
        List<ForkJoinTask<?>> tasks = new ArrayList<>();
        for (TensorSplit split : splits) {
            int chunkStart = (int) split.offset;
            int chunkSize = (int) split.length;
            tasks.add(pool.getUnderlying().submit(() -> action.run(chunkStart, chunkSize)));
        }
        tasks.forEach(ForkJoinTask::join);
    }

    @FunctionalInterface
    private interface RowChunk {
        void run(int chunkStart, int chunkSize);
    }

    private void ensureLocality(AbstractTensor tensor) {
        if (useTensorRuntime() && tensor.locality().isEmpty()) {
            runtime.ensureLocality(tensor);
        }
    }

    private boolean useTensorRuntime() {
        return runtime != null && runMode != RunMode.CALLER_THREAD;
    }
}
