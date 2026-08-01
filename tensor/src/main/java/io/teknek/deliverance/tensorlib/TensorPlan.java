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
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.ForkJoinPool;

/**
 * Small lazy tensor workflow for experimenting with logical tensor plans and local fusion rules.
 *
 * <p>External tensors are introduced with {@link #input(AbstractTensor)} or {@link #mutable(AbstractTensor)}. Operations
 * build a logical graph and execute only when {@link Tensor#materialize()} is called.</p>
 */
public final class TensorPlan {
    private final TensorOperations operations;
    private final WrappedForkJoinPool pool;
    private final MetricRegistry metricRegistry;

    public TensorPlan(TensorOperations operations, WrappedForkJoinPool pool) {
        this(operations, pool, null);
    }

    public TensorPlan(TensorOperations operations, WrappedForkJoinPool pool, MetricRegistry metricRegistry) {
        this.operations = Objects.requireNonNull(operations, "operations");
        this.pool = Objects.requireNonNull(pool, "pool");
        this.metricRegistry = metricRegistry;
    }

    public Tensor input(AbstractTensor tensor) {
        return input("input", tensor);
    }

    public Tensor input(String name, AbstractTensor tensor) {
        return new Tensor(new InputNode(name, tensor, false));
    }

    public ImmutableTensor immutable(String name, AbstractTensor tensor) {
        return new ImmutableTensor(new InputNode(name, tensor, false));
    }

    public Tensor mutable(AbstractTensor tensor) {
        return mutable("mutable", tensor);
    }

    public Tensor mutable(String name, AbstractTensor tensor) {
        return new Tensor(new InputNode(name, tensor, true));
    }

    public FusedBuilder fuse(String name, TensorShape shape) {
        return new FusedBuilder(name, shape);
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
        MUL_IN_PLACE
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
        private final Map<String, Node> inputs = new LinkedHashMap<>();
        private final List<FusedStep> steps = new ArrayList<>();

        private FusedBuilder(String name, TensorShape shape) {
            this.name = Objects.requireNonNull(name, "name");
            this.shape = Objects.requireNonNull(shape, "shape");
        }

        public FusedBuilder read(String name, Tensor tensor) {
            inputs.put(name, tensor.node);
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
            return new Tensor(new FusedNode(name, shape, Map.copyOf(inputs), List.copyOf(steps)));
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
            operations.dotProductChunk(result, a.tensor(), b.tensor(), 0, (int) a.tensor().shape().last(), 0,
                    (int) b.tensor().shape().first());
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
            operations.maccumulate(out, rhs.tensor(), 0, (int) out.shape().last());
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
            operations.accumulate(out, rhs.tensor(), 0, (int) out.shape().last());
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
            operations.scale(factor, out, 0, (int) out.shape().last());
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
            Eval in = input.eval();
            AbstractTensor out = operations.quantize(in.tensor(), dtype, 0, (int) in.tensor().shape().last());
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
        private final Map<String, Node> inputs;
        private final List<FusedStep> steps;

        private FusedNode(String name, TensorShape shape, Map<String, Node> inputs, List<FusedStep> steps) {
            this.name = name;
            this.shape = shape;
            this.inputs = inputs;
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
            AbstractTensor output = new FloatBufferTensor(shape);
            tensors.put(name, output);
            FusedContext context = new FusedContext(tensors);
            long length = shape.size();
            List<TensorSplit> splits = TensorLib.calculateTSplits(0, length, Math.max(1, pool.getCoreCount()));
            List<ForkJoinTask<?>> tasks = new ArrayList<>();
            for (TensorSplit split : splits) {
                tasks.add(pool.getUnderlying().submit(() -> {
                    for (FusedStep step : steps) {
                        Timer.Context timer = startTimer(step.metricName());
                        try {
                            step.map().map(context, split.offset, split.length);
                        } finally {
                            stopTimer(timer);
                        }
                    }
                }));
            }
            tasks.forEach(ForkJoinTask::join);
            evals.values().forEach(TensorPlan::closeIfOwned);
            return new Eval(output, true, true);
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

    private static void closeIfOwned(Eval eval) {
        if (eval.owned()) {
            eval.tensor().close();
        }
    }

    private static AbstractTensor copyOf(AbstractTensor tensor) {
        FloatBufferTensor copy = new FloatBufferTensor(tensor.shape());
        copy.copyFrom(tensor, 0, 0, (int) tensor.size());
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
}
