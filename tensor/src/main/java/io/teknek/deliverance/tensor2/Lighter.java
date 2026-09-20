package io.teknek.deliverance.tensor2;

import com.google.common.base.Preconditions;
import io.dropwizard.metrics5.MetricName;
import io.dropwizard.metrics5.MetricRegistry;
import io.dropwizard.metrics5.Timer;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.dysfx.Either;

import java.util.EnumMap;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.nio.ByteBuffer;
import java.lang.foreign.MemorySegment;
import java.util.concurrent.TimeUnit;

public class Lighter {

    public static final String TENSOR_OP_KEY = "tensor_ops";
    public static final String TENSOR_TYPE = "tensor_type";
    public static final String LENGTH = "length";


    private final Allocator allocator = new Allocator();
    private final MetricRegistry metricRegistry;
    //something like this must be initialized  so later we can pick the right one
    private final EnumMap<TensorProviderKind, TensorOps> tensorOperations = new EnumMap<>(TensorProviderKind.class);

    public record ProviderSelection(Lighter lighter, Map<TensorProviderKind, TensorOps> providers) {
        public ProviderSelection {
            java.util.Objects.requireNonNull(lighter, "lighter");
            providers = Collections.unmodifiableMap(new LinkedHashMap<>(java.util.Objects.requireNonNull(providers,
                    "providers")));
        }
    }

    public record ProviderChoice(Lighter lighter, TensorProviderKind kind) {
        public ProviderChoice {
            java.util.Objects.requireNonNull(lighter, "lighter");
            java.util.Objects.requireNonNull(kind, "kind");
        }
    }

    public Lighter(){
        this(new MetricRegistry());
    }

    public Lighter(MetricRegistry metricRegistry){
        this(metricRegistry, Map.of(
                TensorProviderKind.PANAMA, new PanamaOps(),
                TensorProviderKind.NAIVE, new NaiveOps()
        ));
    }

    public Lighter(MetricRegistry metricRegistry, Map<TensorProviderKind, TensorOps> tensorOperations){
        this.metricRegistry = metricRegistry;
        this.tensorOperations.putAll(tensorOperations);
    }

    public void putTensorOperations(TensorProviderKind kind, TensorOps ops) {
        tensorOperations.put(java.util.Objects.requireNonNull(kind, "kind"),
                java.util.Objects.requireNonNull(ops, "ops"));
    }

    public Map<TensorProviderKind, TensorOps> tensorOperations() {
        return Map.copyOf(tensorOperations);
    }

    public ProviderSelection providersFor(TensorProviderKind... kinds) {
        LinkedHashMap<TensorProviderKind, TensorOps> selected = new LinkedHashMap<>();
        for (TensorProviderKind kind : kinds) {
            TensorOps ops = tensorOperations.get(java.util.Objects.requireNonNull(kind, "kind"));
            if (ops != null) {
                selected.put(kind, ops);
            }
        }
        return new ProviderSelection(this, selected);
    }

    public ProviderChoice providerFor(TensorProviderKind kind) {
        if (!tensorOperations.containsKey(java.util.Objects.requireNonNull(kind, "kind"))) {
            throw new IllegalStateException("No tensor operations registered for " + kind);
        }
        return new ProviderChoice(this, kind);
    }

    public TensorRef allocate(DType dType, TensorShape shape) {
        return allocator.allocate(dType, shape, "cpu");
    }

    public TensorRef allocate(DType dType, TensorShape shape, String device) {
        return allocator.allocate(dType, shape, device);
    }

    /** Copies a mapped raw tensor payload into tensor2-owned storage. */
    public void copyFrom(ByteBuffer source, TensorRef target) {
        Preconditions.checkArgument(source != null, "Source buffer must be set");
        Preconditions.checkArgument(target != null, "Target tensor must be set");
        long bytes = target.underlying().getMemorySegment().byteSize();
        Preconditions.checkArgument(source.remaining() == bytes,
                "Source bytes %s do not match target storage bytes %s", source.remaining(), bytes);
        MemorySegment sourceSegment = MemorySegment.ofBuffer(source.slice());
        target.underlying().getMemorySegment().copyFrom(sourceSegment);
    }

    /** Copies tensor2 storage, including quantization sidecars, between matching tensors. */
    public void copyStorage(TensorRef source, TensorRef target) {
        Preconditions.checkArgument(source != null && target != null, "Source and target tensors must be set");
        Preconditions.checkArgument(source.dType() == target.dType(), "Tensor dtypes must match");
        Preconditions.checkArgument(source.shape().equals(target.shape()), "Tensor shapes must match");
        target.underlying().getMemorySegment().copyFrom(source.underlying().getMemorySegment());
        if (source.dType() == DType.I8) {
            copyStorage(Q8Layout.scale(source), Q8Layout.scale(target));
        } else if (source.dType() == DType.Q4) {
            copyStorage(Q4Layout.scale(source), Q4Layout.scale(target));
        }
    }

    /** Copies a raw F32 scale tensor into a quantized tensor's scale sidecar. */
    public void copyScale(TensorRef sourceScale, TensorRef quantizedTarget) {
        Preconditions.checkArgument(quantizedTarget.dType() == DType.I8 || quantizedTarget.dType() == DType.Q4,
                "Target must be quantized");
        TensorRef targetScale = quantizedTarget.dType() == DType.I8
                ? Q8Layout.scale(quantizedTarget)
                : Q4Layout.scale(quantizedTarget);
        copyStorage(sourceScale, targetScale);
    }

    public boolean shouldQuantizeForEfficiency(TensorRef current, DType desired) {
        Preconditions.checkArgument(current != null, "Current tensor must be set");
        Preconditions.checkArgument(desired != null, "Desired dtype must be set");
        Long currentBytes = allocatedBytes(current.dType(), current.shape());
        Long desiredBytes = allocatedBytes(desired, current.shape());
        return currentBytes != null && desiredBytes != null && desiredBytes < currentBytes;
    }

    public TensorRef reshape(TensorRef input, DType outputDType) {
        Preconditions.checkArgument(input != null, "Input tensor must be set");
        Preconditions.checkArgument(outputDType != null, "Output dtype must be set");
        TensorRef output = allocate(outputDType, input.shape(), input.device());
        for (Map.Entry<TensorProviderKind, TensorOps> entry : tensorOperations.entrySet()) {
            Map<String, String> metricTags = new HashMap<>();
            metricTags.put(TENSOR_OP_KEY, entry.getKey().name());
            metricTags.put("input_type", input.dType().name());
            metricTags.put("output_type", outputDType.name());

            Timer timer = metricRegistry.timer(new MetricName("tensor2.reshape.time", metricTags));
            long startNanos = System.nanoTime();
            Either<OpSupport, Void> result = entry.getValue().reshape(input, output);
            if (result.isRight()) {
                metricRegistry.meter(new MetricName("tensor2.reshape", metricTags)).mark();
                timer.update(System.nanoTime() - startNanos, TimeUnit.NANOSECONDS);
                return output;
            }
        }
        output.close();
        throw new IllegalStateException("No tensor operations support reshape from " + input.dType() + " to "
                + outputDType);
    }

    private Long allocatedBytes(DType dType, TensorShape shape) {
        long values = shape.size();
        return switch (dType) {
            case F32, F16, BF16 -> values * dType.size();
            case I8 -> shape.last() % Q8Layout.BLOCK_SIZE == 0
                    && shape.sparseColumnOffset() % Q8Layout.BLOCK_SIZE == 0
                    && shape.sparseColumnLength() % Q8Layout.BLOCK_SIZE == 0
                    ? values + Q8Layout.scaleShape(shape).size() * DType.F32.size()
                    : null;
            case Q4 -> shape.last() % Q4Layout.BLOCK_SIZE == 0
                    && shape.sparseColumnOffset() % Q4Layout.BLOCK_SIZE == 0
                    && shape.sparseColumnLength() % Q4Layout.BLOCK_SIZE == 0
                    ? values / 2 + Q4Layout.scaleShape(shape).size() * DType.F32.size()
                    : null;
            default -> null;
        };
    }

    public void multiplyAccumulate(MultiplyAccumulate multiplyAccumulate){
        multiplyAccumulate(multiplyAccumulate, Map.of());
    }

    public void multiplyAccumulate(MultiplyAccumulate multiplyAccumulate, Map<String, String> tags){
        TensorRef source = multiplyAccumulate.getSource();
        TensorRef destination = multiplyAccumulate.getDestination();
        Preconditions.checkArgument(destination != null, "Destination tensor must be set");
        Preconditions.checkArgument(source != null, "Source tensor must be set");
        Preconditions.checkArgument(destination.device().equals(source.device()), "Tensors must be on the same device");
        Preconditions.checkArgument(destination.dims() == source.dims(), "Tensors must have the same rank");
        Preconditions.checkArgument(destination.shape().last() == source.shape().last(),
                "Tensors must have the same last dimension");
        Preconditions.checkArgument(source.shape().first() == 1 || destination.shape().first() == source.shape().first(),
                "Source tensor must be broadcastable over destination rows");

        for (Map.Entry<TensorProviderKind, TensorOps> entry : tensorOperations.entrySet()) {
            Map<String, String> metricTags = new HashMap<>(tags);
            metricTags.put(TENSOR_OP_KEY, entry.getKey().name());

            Timer timer = metricRegistry.timer(new MetricName("tensor2.multiply_accumulate.time", metricTags));
            long startNanos = System.nanoTime();
            Either<OpSupport, Void> result = entry.getValue().multiplyAccumulate(destination, source,
                    multiplyAccumulate.getOffset(), multiplyAccumulate.getLength());
            if (result.isRight()) {
                metricRegistry.meter(new MetricName("tensor2.multiply_accumulate", metricTags)).mark();
                timer.update(System.nanoTime() - startNanos, TimeUnit.NANOSECONDS);
                return;
            }
        }
        throw new IllegalStateException("No tensor operations support multiplyAccumulate");
    }

    public void batchDotProduct(BatchDotProduct operation) {
        batchDotProduct(operation, Map.of());
    }

    public void batchDotProduct(BatchDotProduct operation, Map<String, String> tags) {
        TensorRef result = operation.result();
        TensorRef a = operation.a();
        TensorRef b = operation.b();
        Preconditions.checkArgument(result != null, "Result tensor must be set");
        Preconditions.checkArgument(a != null, "A tensor must be set");
        Preconditions.checkArgument(b != null, "B tensor must be set");
        Preconditions.checkArgument(result.device().equals(a.device()) && result.device().equals(b.device()),
                "Tensors must be on the same device");
        Preconditions.checkArgument(result.dType() == DType.F32, "Batch dot output must be F32");
        Preconditions.checkArgument(result.dims() == 2 && a.dims() == 2 && b.dims() == 2,
                "Batch dot requires 2D tensors");
        Preconditions.checkArgument(operation.aRowOffset() >= 0
                && operation.aRowOffset() + result.shape().first() <= a.shape().first(),
                "A row window is out of bounds");
        Preconditions.checkArgument(operation.aColumnOffset() >= 0 && operation.bColumnOffset() >= 0
                && operation.columnLength() >= 0
                && operation.aColumnOffset() + operation.columnLength() <= a.shape().last()
                && operation.bColumnOffset() + operation.columnLength() <= b.shape().last(),
                "Column window is out of bounds");
        Preconditions.checkArgument(operation.bRowOffset() >= 0 && operation.rowChunkSize() >= 0
                && operation.bRowOffset() + operation.rowChunkSize() <= b.shape().first(),
                "B row window is out of bounds");
        Preconditions.checkArgument(operation.resultRowOffset() >= 0
                && operation.resultRowOffset() + operation.bRowOffset() + operation.rowChunkSize()
                <= result.shape().last(), "Result row offset is out of bounds");

        for (Map.Entry<TensorProviderKind, TensorOps> entry : tensorOperations.entrySet()) {
            Map<String, String> metricTags = new HashMap<>(tags);
            metricTags.put(TENSOR_OP_KEY, entry.getKey().name());

            Timer timer = metricRegistry.timer(new MetricName("tensor2.batch_dot_product.time", metricTags));
            long startNanos = System.nanoTime();
            Either<OpSupport, Void> outcome = entry.getValue().batchDotProduct(operation);
            if (outcome.isRight()) {
                metricRegistry.meter(new MetricName("tensor2.batch_dot_product", metricTags)).mark();
                timer.update(System.nanoTime() - startNanos, TimeUnit.NANOSECONDS);
                return;
            }
        }
        throw new IllegalStateException("No tensor operations support batchDotProduct");
    }

    public void scale(Scale scale) {
        scale(scale, Map.of());
    }

    public void scale(Scale scale, Map<String, String> tags) {
        validateScale(scale);
        for (Map.Entry<TensorProviderKind, TensorOps> entry : tensorOperations.entrySet()) {
            if (scaleWithProvider(scale, tags, entry.getKey(), entry.getValue())) {
                return;
            }
        }
        throw new IllegalStateException("No tensor operations support scale");
    }

    public boolean scale(Scale scale, Map<String, String> tags, TensorProviderKind kind) {
        validateScale(scale);
        TensorOps ops = tensorOperations.get(java.util.Objects.requireNonNull(kind, "kind"));
        if (ops == null) {
            throw new IllegalStateException("No tensor operations registered for " + kind);
        }
        return scaleWithProvider(scale, tags, kind, ops);
    }

    private void validateScale(Scale scale) {
        TensorRef target = scale.getTarget();
        Preconditions.checkArgument(target != null, "Target tensor must be set");
        Preconditions.checkArgument(scale.getOffset() >= 0 && scale.getLength() >= 0
                && scale.getOffset() + scale.getLength() <= target.shape().last(), "Scale window out of bounds");
    }

    private boolean scaleWithProvider(Scale scale, Map<String, String> tags, TensorProviderKind kind, TensorOps ops) {
        TensorRef target = scale.getTarget();
        Map<String, String> metricTags = new HashMap<>(tags);
        metricTags.put(TENSOR_OP_KEY, kind.name());
        metricTags.put(LENGTH, String.valueOf(scale.getLength()));
        metricTags.put(TENSOR_TYPE, scale.getTarget().dType().name());
        Timer timer = metricRegistry.timer(new MetricName("tensor2.scale.time", metricTags));
        long startNanos = System.nanoTime();
        Either<OpSupport, Void> result = ops.scale(scale.getFactor(), target, scale.getOffset(), scale.getLength());
        if (result.isRight()) {
            metricRegistry.meter(new MetricName("tensor2.scale", metricTags)).mark();
            timer.update(System.nanoTime() - startNanos, TimeUnit.NANOSECONDS);
            return true;
        }
        return false;
    }

    public TensorRef to(TensorRef a, String device){
        //ask the allocator to do this
        return null;
    }
}
