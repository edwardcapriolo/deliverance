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
import java.util.Objects;
import java.nio.ByteBuffer;
import java.lang.foreign.MemorySegment;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

public class Lighter {

    public static final String TENSOR_OP_KEY = "tensor_ops";
    public static final String TENSOR_TYPE = "tensor_type";
    public static final String LENGTH = "length";


    private final Allocator allocator = new Allocator();
    private final MetricRegistry metricRegistry;
    //something like this must be initialized  so later we can pick the right one
    private final EnumMap<TensorProviderKind, TensorOps> tensorOperations = new EnumMap<>(TensorProviderKind.class);
    private final EnumMap<TensorProviderKind, CompositeOpsProvider> compositeOperations =
            new EnumMap<>(TensorProviderKind.class);
    private volatile Consumer<ProviderEvent> providerObserver = ignored -> { };

    public record ProviderEvent(String operation, TensorProviderKind kind, boolean supported,
            DType outputType, DType inputType, DType weightType, Map<String, String> tags) {
        public ProviderEvent {
            java.util.Objects.requireNonNull(operation, "operation");
            java.util.Objects.requireNonNull(kind, "kind");
            tags = Collections.unmodifiableMap(new LinkedHashMap<>(java.util.Objects.requireNonNull(tags, "tags")));
        }
    }

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
        if (ops instanceof CompositeOpsProvider composite) {
            compositeOperations.put(kind, composite);
        } else {
            compositeOperations.remove(kind);
        }
    }

    /** Controls whether allocateDirty may return pooled storage without zeroing it. */
    public void allowDirtyTensors(boolean allowDirtyTensors) {
        allocator.allowDirtyTensors(allowDirtyTensors);
    }

    public void setProviderObserver(Consumer<ProviderEvent> providerObserver) {
        this.providerObserver = java.util.Objects.requireNonNull(providerObserver, "providerObserver");
    }

    public Map<TensorProviderKind, TensorOps> tensorOperations() {
        return Map.copyOf(tensorOperations);
    }

    public Map<TensorProviderKind, CompositeOpsProvider> compositeOperationProviders() {
        return Map.copyOf(compositeOperations);
    }

    /** Returns the preferred CPU provider for a composite operation. */
    public TensorOps preferredTensorOperations() {
        TensorOps ops = tensorOperations.get(TensorProviderKind.SIMD);
        if (ops != null) return ops;
        ops = tensorOperations.get(TensorProviderKind.PANAMA);
        if (ops != null) return ops;
        ops = tensorOperations.get(TensorProviderKind.NAIVE);
        if (ops != null) return ops;
        throw new IllegalStateException("No CPU tensor operations registered");
    }

    /** Providers in the same priority order used by the composite fallback dispatch. */
    public java.util.List<TensorOps> orderedTensorOperations() {
        java.util.ArrayList<TensorOps> result = new java.util.ArrayList<>();
        for (TensorProviderKind kind : TensorProviderKind.values()) {
            TensorOps ops = tensorOperations.get(kind);
            if (ops != null) result.add(ops);
        }
        return result;
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

    public TensorRef allocateDirty(DType dType, TensorShape shape) {
        return allocator.allocateDirty(dType, shape);
    }

    /** Clears tensor payload and quantization sidecar storage. */
    public void clear(TensorRef target) {
        Preconditions.checkArgument(target != null, "Target tensor must be set");
        allocator.clear(target);
    }

    /** Materializes a dense tensor with its dimensions and coordinates reversed. */
    public void transpose(Transpose transpose) {
        TensorRef source = transpose.getSource();
        TensorRef destination = transpose.getDestination();
        Preconditions.checkArgument(source != null, "Source tensor must be set");
        Preconditions.checkArgument(destination != null, "Destination tensor must be set");
        Preconditions.checkArgument(source.dType() == destination.dType(), "Tensor dtypes must match");
        Preconditions.checkArgument(!source.shape().isSparse() && !destination.shape().isSparse(),
                "Cannot transpose sparse tensors");
        Preconditions.checkArgument(source.dims() == destination.dims(), "Tensor ranks must match");
        for (int dimension = 0; dimension < source.dims(); dimension++) {
            Preconditions.checkArgument(destination.shape().dim(dimension)
                            == source.shape().dim(source.dims() - dimension - 1),
                    "Destination shape must reverse source shape");
        }
        int[] cursor = new int[source.dims()];
        do {
            int[] destinationCursor = new int[cursor.length];
            for (int dimension = 0; dimension < cursor.length; dimension++) {
                destinationCursor[dimension] = cursor[cursor.length - dimension - 1];
            }
            destination.set(source.get(cursor), destinationCursor);
        } while (advance(cursor, source.shape()));
    }

    /** Materializes equal-sized logical chunks into caller-owned destination tensors. */
    public void split(Split split) {
        TensorRef source = split.getSource();
        TensorRef[] destinations = split.getDestinations();
        int chunks = split.getChunks();
        int dimension = split.getDimension();
        Preconditions.checkArgument(source != null, "Source tensor must be set");
        Preconditions.checkArgument(destinations != null, "Destination tensors must be set");
        Preconditions.checkArgument(!source.shape().isSparse(), "Cannot split sparse tensors");
        Preconditions.checkArgument(chunks > 0 && destinations.length == chunks,
                "Destination count must equal chunk count");
        Preconditions.checkArgument(dimension >= 0 && dimension < source.dims(), "Split dimension out of bounds");
        int dimensionSize = source.shape().dim(dimension);
        Preconditions.checkArgument(dimensionSize % chunks == 0, "Chunks must be of equal size");
        int[] expectedShape = source.shape().shapeArray();
        expectedShape[dimension] = dimensionSize / chunks;
        for (TensorRef destination : destinations) {
            Preconditions.checkArgument(destination != null, "Destination tensor must be set");
            Preconditions.checkArgument(destination.dType() == source.dType(), "Tensor dtypes must match");
            Preconditions.checkArgument(!destination.shape().isSparse(), "Cannot split into sparse tensors");
            Preconditions.checkArgument(java.util.Arrays.equals(destination.shape().shapeArray(), expectedShape),
                    "Destination shape does not match split shape");
        }

        for (int chunk = 0; chunk < chunks; chunk++) {
            int[] localCursor = new int[source.dims()];
            do {
                int[] sourceCursor = localCursor.clone();
                sourceCursor[dimension] += chunk * expectedShape[dimension];
                destinations[chunk].set(source.get(sourceCursor), localCursor);
            } while (advance(localCursor, TensorShape.of(expectedShape)));
        }
    }

    private boolean advance(int[] cursor, TensorShape shape) {
        for (int dimension = cursor.length - 1; dimension >= 0; dimension--) {
            if (++cursor[dimension] < shape.dim(dimension)) {
                return true;
            }
            cursor[dimension] = 0;
        }
        return false;
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

    /** Copies logical elements using the same offset/length contract as {@code AbstractTensor.copyFrom}. */
    public void copyFrom(TensorRef source, TensorRef target, int sourceOffset, int targetOffset, int length) {
        Preconditions.checkArgument(source != null && target != null, "Source and target tensors must be set");
        Preconditions.checkArgument(source.dType() == target.dType(), "Tensor dtypes must match");
        Preconditions.checkArgument(sourceOffset >= 0 && targetOffset >= 0 && length >= 0,
                "Copy offsets and length must be non-negative");
        Preconditions.checkArgument((long) sourceOffset + length <= source.shape().size(),
                "Source copy range out of bounds");
        Preconditions.checkArgument((long) targetOffset + length <= target.shape().size(),
                "Target copy range out of bounds");
        long bytes = source.dType() == DType.Q4
                ? length / 2L
                : (long) length * source.dType().size();
        target.memorySegment().asSlice(target.memorySegmentOffset(targetOffset), bytes)
                .copyFrom(source.memorySegment().asSlice(source.memorySegmentOffset(sourceOffset), bytes));
        if (source.dType() == DType.I8) {
            copyScaleRange(Q8Layout.scale(source), Q8Layout.scale(target), sourceOffset, targetOffset,
                    length, Q8Layout.BLOCK_SIZE);
        } else if (source.dType() == DType.Q4) {
            copyScaleRange(Q4Layout.scale(source), Q4Layout.scale(target), sourceOffset, targetOffset,
                    length, Q4Layout.BLOCK_SIZE);
        }
    }

    private void copyScaleRange(TensorRef source, TensorRef target, int sourceOffset, int targetOffset,
            int length, int blockSize) {
        Preconditions.checkArgument(source != null && target != null, "Quantized scale sidecars are required");
        int sourceScaleOffset = sourceOffset / blockSize;
        int targetScaleOffset = targetOffset / blockSize;
        int scaleLength = length / blockSize;
        long bytes = (long) scaleLength * DType.F32.size();
        target.memorySegment().asSlice(target.memorySegmentOffset(targetScaleOffset), bytes)
                .copyFrom(source.memorySegment().asSlice(source.memorySegmentOffset(sourceScaleOffset), bytes));
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

    /**
     * Converts {@code input} into a newly allocated tensor of {@code outputDType}.
     *
     * <p>This is a convenience method, not an in-place conversion. Hot paths should generally
     * allocate their final destination and use {@link #reshape(TensorRef, TensorRef)} to avoid an
     * intermediate tensor and a subsequent copy.</p>
     */
    public TensorRef reshape(TensorRef input, DType outputDType) {
        Preconditions.checkArgument(input != null, "Input tensor must be set");
        Preconditions.checkArgument(outputDType != null, "Output dtype must be set");
        TensorRef output = allocate(outputDType, input.shape(), input.device());
        try {
            reshape(input, output);
            return output;
        } catch (RuntimeException e) {
            output.close();
            throw e;
        }
    }

    /**
     * Converts {@code input} directly into the caller-owned {@code output} tensor.
     *
     * <p>Prefer this overload when the destination is already known, especially in model
     * execution loops. It avoids the intermediate allocation performed by
     * {@link #reshape(TensorRef, DType)}.</p>
     */
    public void reshape(TensorRef input, TensorRef output) {
        Preconditions.checkArgument(input != null, "Input tensor must be set");
        Preconditions.checkArgument(output != null, "Output tensor must be set");
        Preconditions.checkArgument(input.shape().equals(output.shape()), "Input and output shapes must match");
        Preconditions.checkArgument(input.device().equals(output.device()), "Input and output devices must match");
        for (Map.Entry<TensorProviderKind, TensorOps> entry : tensorOperations.entrySet()) {
            Map<String, String> metricTags = new HashMap<>();
            metricTags.put(TENSOR_OP_KEY, entry.getKey().name());
            metricTags.put("input_type", input.dType().name());
            metricTags.put("output_type", output.dType().name());

            Timer timer = metricRegistry.timer(new MetricName("tensor2.reshape.time", metricTags));
            long startNanos = System.nanoTime();
            Either<OpSupport, Void> result = entry.getValue().reshape(input, output);
            if (result.isRight()) {
                metricRegistry.meter(new MetricName("tensor2.reshape", metricTags)).mark();
                timer.update(System.nanoTime() - startNanos, TimeUnit.NANOSECONDS);
                observe("reshape", entry.getKey(), true, output.dType(), input.dType(), null, Map.of());
                return;
            }
            metricRegistry.meter(new MetricName("tensor2.reshape.unsupported", metricTags)).mark();
            observe("reshape", entry.getKey(), false, output.dType(), input.dType(), null, Map.of());
        }
        throw new IllegalStateException("No tensor operations support reshape from " + input.dType() + " to "
                + output.dType());
    }

    public void copy(TensorRef source, int sourceOffset, TensorRef destination, int destinationOffset, int length) {
        Preconditions.checkArgument(source != null, "Source tensor must be set");
        Preconditions.checkArgument(destination != null, "Destination tensor must be set");
        Preconditions.checkArgument(source.dType() == destination.dType(), "Source and destination dtype must match");
        Preconditions.checkArgument(sourceOffset >= 0 && destinationOffset >= 0 && length >= 0,
                "Copy offsets and length must be non-negative");
        Preconditions.checkArgument(sourceOffset + length <= source.shape().size(), "Source copy range out of bounds");
        Preconditions.checkArgument(destinationOffset + length <= destination.shape().size(),
                "Destination copy range out of bounds");
        long bytes = switch (source.dType()) {
            case Q4 -> {
                Preconditions.checkArgument(sourceOffset % 2 == 0 && destinationOffset % 2 == 0 && length % 2 == 0,
                        "Q4 copy ranges must be byte-aligned");
                yield length / 2L;
            }
            default -> (long) length * source.dType().size();
        };
        destination.memorySegment().asSlice(destination.memorySegmentOffset(destinationOffset), bytes)
                .copyFrom(source.memorySegment().asSlice(source.memorySegmentOffset(sourceOffset), bytes));
        if (source.dType() == DType.I8) {
            copyQuantizedScale(Q8Layout.scale(source), Q8Layout.scale(destination), Q8Layout.BLOCK_SIZE,
                    sourceOffset, destinationOffset, length);
        } else if (source.dType() == DType.Q4) {
            copyQuantizedScale(Q4Layout.scale(source), Q4Layout.scale(destination), Q4Layout.BLOCK_SIZE,
                    sourceOffset, destinationOffset, length);
        }
    }

    private void copyQuantizedScale(TensorRef sourceScale, TensorRef destinationScale, int blockSize,
            int sourceOffset, int destinationOffset, int length) {
        if (sourceScale == null || destinationScale == null) {
            return;
        }
        Preconditions.checkArgument(sourceOffset % blockSize == 0 && destinationOffset % blockSize == 0
                && length % blockSize == 0, "Quantized scale copy ranges must be block-aligned");
        int sourceScaleOffset = sourceOffset / blockSize;
        int destinationScaleOffset = destinationOffset / blockSize;
        int scaleLength = length / blockSize;
        destinationScale.memorySegment().asSlice(destinationScale.memorySegmentOffset(destinationScaleOffset),
                (long) scaleLength * DType.F32.size())
                .copyFrom(sourceScale.memorySegment().asSlice(sourceScale.memorySegmentOffset(sourceScaleOffset),
                        (long) scaleLength * DType.F32.size()));
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

    public void accumulate(Accumulate accumulate) {
        accumulate(accumulate, Map.of());
    }

    public void argMax(ArgMax argMax) {
        TensorRef input = argMax.getInput();
        TensorRef output = argMax.getOutput();
        Preconditions.checkArgument(input != null, "Input tensor must be set");
        Preconditions.checkArgument(output != null, "Output tensor must be set");
        for (Map.Entry<TensorProviderKind, TensorOps> entry : tensorOperations.entrySet()) {
            Either<OpSupport, Void> result = entry.getValue().argMax(input, output,
                    argMax.getOffset(), argMax.getLength());
            if (result.isRight()) {
                return;
            }
        }
        throw new IllegalStateException("No tensor operations support argMax");
    }

    public void max(Max max) {
        TensorRef input = max.getSource();
        TensorRef output = max.getDestination();
        Preconditions.checkArgument(input != null, "Source tensor must be set");
        Preconditions.checkArgument(output != null, "Destination tensor must be set");
        for (Map.Entry<TensorProviderKind, TensorOps> entry : tensorOperations.entrySet()) {
            Either<OpSupport, Void> result = entry.getValue().max(input, max.getRow(), max.getOffset(),
                    max.getLength(), output);
            if (result.isRight()) {
                return;
            }
        }
        throw new IllegalStateException("No tensor operations support max");
    }

    public void exp(Exp exp) {
        TensorRef input = exp.getSource();
        TensorRef output = exp.getDestination();
        Preconditions.checkArgument(input != null, "Source tensor must be set");
        Preconditions.checkArgument(output != null, "Destination tensor must be set");
        for (Map.Entry<TensorProviderKind, TensorOps> entry : tensorOperations.entrySet()) {
            Either<OpSupport, Void> result = entry.getValue().exp(input, output, exp.getOffset(), exp.getLength());
            if (result.isRight()) {
                return;
            }
        }
        throw new IllegalStateException("No tensor operations support exp");
    }

    public void sum(Sum sum) {
        TensorRef input = sum.getSource();
        TensorRef output = sum.getDestination();
        Preconditions.checkArgument(input != null, "Source tensor must be set");
        Preconditions.checkArgument(output != null, "Destination tensor must be set");
        for (Map.Entry<TensorProviderKind, TensorOps> entry : tensorOperations.entrySet()) {
            Either<OpSupport, Void> result = entry.getValue().sum(input, sum.getRow(), sum.getOffset(),
                    sum.getLength(), output);
            if (result.isRight()) {
                return;
            }
        }
        throw new IllegalStateException("No tensor operations support sum");
    }

    public void accumulate(Accumulate accumulate, Map<String, String> tags) {
        TensorRef source = accumulate.getSource();
        TensorRef destination = accumulate.getDestination();
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
            Timer timer = metricRegistry.timer(new MetricName("tensor2.accumulate.time", metricTags));
            long startNanos = System.nanoTime();
            Either<OpSupport, Void> result = entry.getValue().accumulate(destination, source,
                    accumulate.getOffset(), accumulate.getLength());
            if (result.isRight()) {
                metricRegistry.meter(new MetricName("tensor2.accumulate", metricTags)).mark();
                timer.update(System.nanoTime() - startNanos, TimeUnit.NANOSECONDS);
                observe("accumulate", entry.getKey(), true, destination.dType(), destination.dType(),
                        source.dType(), tags);
                return;
            }
            metricRegistry.meter(new MetricName("tensor2.accumulate.unsupported", metricTags)).mark();
            observe("accumulate", entry.getKey(), false, destination.dType(), destination.dType(),
                    source.dType(), tags);
        }
        throw new IllegalStateException("No tensor operations support accumulate");
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
                observe("multiply_accumulate", entry.getKey(), true, destination.dType(), destination.dType(),
                        source.dType(), tags);
                return;
            }
            metricRegistry.meter(new MetricName("tensor2.multiply_accumulate.unsupported", metricTags)).mark();
            observe("multiply_accumulate", entry.getKey(), false, destination.dType(), destination.dType(),
                    source.dType(), tags);
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
                observe("batch_dot_product", entry.getKey(), true, result.dType(), a.dType(), b.dType(), tags);
                return;
            }
            metricRegistry.meter(new MetricName("tensor2.batch_dot_product.unsupported", metricTags)).mark();
            observe("batch_dot_product", entry.getKey(), false, result.dType(), a.dType(), b.dType(), tags);
        }
        throw new IllegalStateException("No tensor operations support batchDotProduct");
    }

    public void dotProductRows(TensorRef output, TensorRef input, TensorRef weights,
            int inputStart, int inputLength, int weightRowStart, int weightRowCount, int outputColumnStart) {
        dotProductRows(output, input, weights, inputStart, inputStart, inputLength, weightRowStart,
                weightRowCount, outputColumnStart, Map.of());
    }

    public void dotProductRows(TensorRef output, TensorRef input, TensorRef weights,
            int inputColumnStart, int weightColumnStart, int columnLength, int weightRowStart,
            int weightRowCount, int outputColumnStart) {
        dotProductRows(output, input, weights, inputColumnStart, weightColumnStart, columnLength, weightRowStart,
                weightRowCount, outputColumnStart, Map.of());
    }

    public void dotProductRows(TensorRef output, TensorRef input, TensorRef weights,
            int inputColumnStart, int weightColumnStart, int columnLength, int weightRowStart,
            int weightRowCount, int outputColumnStart, Map<String, String> tags) {
        Preconditions.checkArgument(output != null, "Output tensor must be set");
        Preconditions.checkArgument(input != null, "Input tensor must be set");
        Preconditions.checkArgument(weights != null, "Weights tensor must be set");
        Preconditions.checkArgument(output.dims() == 2 && input.dims() == 2 && weights.dims() == 2,
                "dotProductRows requires 2D tensors");
        Preconditions.checkArgument(output.shape().first() == input.shape().first(),
                "Output and input row counts must match");
        Preconditions.checkArgument(inputColumnStart >= 0 && weightColumnStart >= 0 && columnLength >= 0
                        && inputColumnStart + columnLength <= input.shape().last()
                        && weightColumnStart + columnLength <= weights.shape().last(),
                "Input range is out of bounds");
        Preconditions.checkArgument(weightRowStart >= 0 && weightRowCount >= 0
                        && weightRowStart + weightRowCount <= weights.shape().first(),
                "Weight row range is out of bounds");
        Preconditions.checkArgument(outputColumnStart >= 0
                        && outputColumnStart + weightRowCount <= output.shape().last(),
                "Output column range is out of bounds");

        for (Map.Entry<TensorProviderKind, TensorOps> entry : tensorOperations.entrySet()) {
            Map<String, String> metricTags = new HashMap<>(tags);
            metricTags.put(TENSOR_OP_KEY, entry.getKey().name());
            Timer timer = metricRegistry.timer(new MetricName("tensor2.dot_product_rows.time", metricTags));
            long startNanos = System.nanoTime();
            Either<OpSupport, Void> outcome = entry.getValue().dotProductRows(output, input, weights,
                    inputColumnStart, weightColumnStart, columnLength, weightRowStart, weightRowCount,
                    outputColumnStart);
            if (outcome.isRight()) {
                metricRegistry.meter(new MetricName("tensor2.dot_product_rows", metricTags)).mark();
                timer.update(System.nanoTime() - startNanos, TimeUnit.NANOSECONDS);
                observe("dot_product_rows", entry.getKey(), true, output.dType(), input.dType(), weights.dType(), tags);
                return;
            }
            metricRegistry.meter(new MetricName("tensor2.dot_product_rows.unsupported", metricTags)).mark();
            observe("dot_product_rows", entry.getKey(), false, output.dType(), input.dType(), weights.dType(), tags);
        }
        throw new IllegalStateException("No tensor operations support dotProductRows");
    }

    public void dotProductBatchChunk(DotProductBatchChunk operation) {
        Objects.requireNonNull(operation, "operation");
        TensorRef[] results = operation.results();
        TensorRef[] weights = operation.weights();
        Preconditions.checkArgument(results != null && weights != null && results.length == weights.length
                && results.length > 0, "Paired projection results and weights must have equal non-zero lengths");
        Preconditions.checkArgument(operation.input() != null, "Paired projection input must be set");
        for (int i = 0; i < results.length; i++) {
            Preconditions.checkArgument(results[i] != null && weights[i] != null,
                    "Paired projection tensors must be set");
        }
        for (Map.Entry<TensorProviderKind, TensorOps> entry : tensorOperations.entrySet()) {
            Either<OpSupport, Void> outcome = entry.getValue().dotProductBatchChunk(operation);
            if (outcome.isRight()) {
                observe("dot_product_batch_chunk", entry.getKey(), true, results[0].dType(),
                        operation.input().dType(), weights[0].dType(), Map.of());
                return;
            }
            observe("dot_product_batch_chunk", entry.getKey(), false, results[0].dType(),
                    operation.input().dType(), weights[0].dType(), Map.of());
        }
        throw new UnsupportedOperationException("No tensor operations support dotProductBatchChunk");
    }

    public boolean supportsDotProductBatchChunk(DotProductBatchChunk operation) {
        for (TensorOps ops : tensorOperations.values()) {
            if (ops.supportsDotProductBatchChunk(operation)) {
                return true;
            }
        }
        return false;
    }

    Either<OpSupport, Void> activationMultiplyQuantize(ActivationMultiplyQuantize operation) {
        for (CompositeOpsProvider provider : compositeOperations.values()) {
            Either<OpSupport, Void> outcome = provider.activationMultiplyQuantize(operation);
            if (outcome.isRight()) {
                return outcome;
            }
        }
        return Either.Left(OpSupport.Unsupported);
    }

    boolean supportsActivationMultiplyQuantize(ActivationMultiplyQuantize operation) {
        for (CompositeOpsProvider provider : compositeOperations.values()) {
            if (provider.supportsActivationMultiplyQuantize(operation)) {
                return true;
            }
        }
        return false;
    }

    public void saxpy(float alpha, TensorRef x, TensorRef y, int xOffset, int yOffset, int length) {
        Preconditions.checkArgument(x != null && y != null, "SAXPY tensors must be set");
        Preconditions.checkArgument(x.dims() == 2 && y.dims() == 2 && y.shape().first() == 1,
                "SAXPY requires 2D x and one-row y tensors");
        Preconditions.checkArgument(xOffset >= 0 && yOffset >= 0 && length >= 0
                        && xOffset + length <= x.shape().last()
                        && yOffset + length <= y.shape().last(),
                "SAXPY range is out of bounds");
        for (Map.Entry<TensorProviderKind, TensorOps> entry : tensorOperations.entrySet()) {
            Map<String, String> metricTags = new HashMap<>();
            metricTags.put(TENSOR_OP_KEY, entry.getKey().name());
            metricTags.put("mode", "scalar");
            Timer timer = metricRegistry.timer(new MetricName("tensor2.saxpy.time", metricTags));
            long startNanos = System.nanoTime();
            if (entry.getValue().saxpy(alpha, x, y, xOffset, yOffset, length).isRight()) {
                metricRegistry.meter(new MetricName("tensor2.saxpy", metricTags)).mark();
                timer.update(System.nanoTime() - startNanos, TimeUnit.NANOSECONDS);
                observe("saxpy", entry.getKey(), true, y.dType(), x.dType(), null, Map.of("mode", "scalar"));
                return;
            }
            metricRegistry.meter(new MetricName("tensor2.saxpy.unsupported", metricTags)).mark();
            observe("saxpy", entry.getKey(), false, y.dType(), x.dType(), null, Map.of("mode", "scalar"));
        }
        throw new IllegalStateException("No tensor operations support saxpy");
    }

    public void saxpy(TensorRef alpha, TensorRef x, TensorRef y, int xOffset, int yOffset, int length,
            int alphaOffset, int xRowOffset, int batchSize) {
        Preconditions.checkArgument(alpha != null && x != null && y != null, "SAXPY tensors must be set");
        Preconditions.checkArgument(alpha.dims() == 2 && x.dims() == 2 && y.dims() == 2
                        && y.shape().first() == 1,
                "Batch SAXPY requires 2D tensors and one-row y");
        Preconditions.checkArgument(alphaOffset >= 0 && batchSize >= 0
                        && alphaOffset + batchSize <= alpha.shape().last(),
                "SAXPY alpha range is out of bounds");
        Preconditions.checkArgument(xRowOffset >= 0 && xRowOffset + batchSize <= x.shape().first(),
                "SAXPY x row range is out of bounds");
        Preconditions.checkArgument(xOffset >= 0 && yOffset >= 0 && length >= 0
                        && xOffset + length <= x.shape().last()
                        && yOffset + length <= y.shape().last(),
                "SAXPY range is out of bounds");
        for (Map.Entry<TensorProviderKind, TensorOps> entry : tensorOperations.entrySet()) {
            Map<String, String> metricTags = new HashMap<>();
            metricTags.put(TENSOR_OP_KEY, entry.getKey().name());
            metricTags.put("mode", "batch");
            Timer timer = metricRegistry.timer(new MetricName("tensor2.saxpy.time", metricTags));
            long startNanos = System.nanoTime();
            if (entry.getValue().saxpy(alpha, x, y, xOffset, yOffset, length, alphaOffset, xRowOffset,
                    batchSize).isRight()) {
                metricRegistry.meter(new MetricName("tensor2.saxpy", metricTags)).mark();
                timer.update(System.nanoTime() - startNanos, TimeUnit.NANOSECONDS);
                observe("saxpy", entry.getKey(), true, y.dType(), x.dType(), alpha.dType(), Map.of("mode", "batch"));
                return;
            }
            metricRegistry.meter(new MetricName("tensor2.saxpy.unsupported", metricTags)).mark();
            observe("saxpy", entry.getKey(), false, y.dType(), x.dType(), alpha.dType(), Map.of("mode", "batch"));
        }
        throw new IllegalStateException("No tensor operations support batch saxpy");
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
            observe("scale", kind, true, target.dType(), target.dType(), null, tags);
            return true;
        }
        observe("scale", kind, false, target.dType(), target.dType(), null, tags);
        return false;
    }

    private void observe(String operation, TensorProviderKind kind, boolean supported, DType outputType,
            DType inputType, DType weightType, Map<String, String> tags) {
        providerObserver.accept(new ProviderEvent(operation, kind, supported, outputType, inputType, weightType, tags));
    }

    public TensorRef to(TensorRef a, String device){
        //ask the allocator to do this
        return null;
    }
}
