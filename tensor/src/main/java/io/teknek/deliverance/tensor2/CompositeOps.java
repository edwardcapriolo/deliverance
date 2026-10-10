package io.teknek.deliverance.tensor2;

import com.google.common.base.Preconditions;
import io.dropwizard.metrics5.MetricName;
import io.dropwizard.metrics5.MetricRegistry;
import io.dropwizard.metrics5.Timer;
import net.jafama.FastMath;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

public final class CompositeOps {
    private static final VectorSpecies<Float> F32_SPECIES = FloatVector.SPECIES_PREFERRED;
    public static final String TENSOR_TYPE = Lighter.TENSOR_TYPE;
    public static final String LENGTH = Lighter.LENGTH;

    private final Lighter lighter;
    private final MetricRegistry metricRegistry;

    public CompositeOps(Lighter lighter, MetricRegistry metricRegistry) {
        this.lighter = Objects.requireNonNull(lighter, "lighter");
        this.metricRegistry = metricRegistry;
    }

    public CompositeOps(Lighter lighter) {
        this(lighter, null);
    }

    public void multiplyInPlace(MultiplyInPlace operation) {
        multiplyInPlace(operation, Map.of());
    }

    public void multiplyInPlace(MultiplyInPlace operation, Map<String, String> tags) {
        validateMultiplyInPlace(operation);
        run("tensor2.composite.multiply_in_place", operation.getTarget(), operation.getLength(), tags, () -> {
            try {
                lighter.scale(new Scale(operation.getFactor())
                        .target(operation.getTarget())
                        .offsetAndLength(operation.getOffset(), operation.getLength()), tags);
            } catch (IllegalStateException e) {
                throw new UnsupportedOperationException("No tensor operations support multiplyInPlace", e);
            }
        });
    }

    /** Runs the provider-level paired projection used by the legacy gate/up fast path. */
    public void dotProductBatchChunk(DotProductBatchChunk operation) {
        lighter.dotProductBatchChunk(operation);
    }

    public TensorRef activationMultiplyQuantize(ActivationMultiplyQuantize operation) {
        Objects.requireNonNull(operation, "operation");
        TensorRef output = operation.output();
        boolean owned = false;
        if (output == null) {
            output = lighter.allocate(operation.qtype(), operation.gate().shape());
            operation.output(output);
            owned = true;
        }
        try {
            if (lighter.activationMultiplyQuantize(operation).isLeft()) {
                throw new UnsupportedOperationException("No composite provider supports activationMultiplyQuantize");
            }
            return output;
        } catch (RuntimeException | Error e) {
            if (owned) {
                output.close();
            }
            throw e;
        }
    }

    public boolean supportsActivationMultiplyQuantize(ActivationMultiplyQuantize operation) {
        return lighter.supportsActivationMultiplyQuantize(operation);
    }

    /** Generic one-token causal attention over paged KV tensors. */
    public void decodePagedAttention(TensorRef output, TensorRef query, TensorRef[] keyPages,
            TensorRef[] valuePages, int visibleRows, int numberOfHeads, int numberOfKeyValueHeads, int headSize,
            float scale, Float softcap, ForkJoinPool pool, int headSplitSize) {
        Objects.requireNonNull(pool, "pool");
        Preconditions.checkArgument(keyPages.length == valuePages.length, "key/value page count mismatch");
        Preconditions.checkArgument(query.shape().first() == 1 && output.shape().first() == 1,
                "paged decode attention expects one query row");
        Preconditions.checkArgument(numberOfKeyValueHeads > 0 && numberOfHeads % numberOfKeyValueHeads == 0,
                "GQA heads must divide evenly");
        DecodePagedAttention operation = new DecodePagedAttention(output, query, keyPages, valuePages, visibleRows,
                numberOfHeads, numberOfKeyValueHeads, headSize, scale, softcap, pool, headSplitSize);
        for (CompositeOpsProvider provider : lighter.compositeOperationProviders().values()) {
            if (provider.supportsDecodePagedAttention(operation)) {
                if (provider.decodePagedAttention(operation).isRight()) {
                    return;
                }
            }
        }
        decodePagedAttentionFallback(operation);
    }

    private void decodePagedAttentionFallback(DecodePagedAttention operation) {
        TensorRef output = operation.output();
        TensorRef query = operation.query();
        TensorRef[] keyPages = operation.keyPages();
        TensorRef[] valuePages = operation.valuePages();
        int visibleRows = operation.visibleRows();
        int numberOfHeads = operation.numberOfHeads();
        int numberOfKeyValueHeads = operation.numberOfKeyValueHeads();
        int headSize = operation.headSize();
        float scale = operation.scale();
        Float softcap = operation.softcap();
        ForkJoinPool pool = operation.pool();
        int headSplitSize = operation.headSplitSize();
        int headGroupSize = numberOfHeads / numberOfKeyValueHeads;
        output.memorySegment().fill((byte) 0);
        java.util.List<TensorOps> ops = lighter.orderedTensorOperations();
        // Match the legacy VectorMath.pfor scheduling: one logical task per head,
        // executed as a parallel stream inside the supplied model pool.
        pool.submit(() -> IntStream.range(0, numberOfHeads).parallel().forEach(head ->
                decodePagedAttentionHead(output, query, keyPages, valuePages, visibleRows,
                        headGroupSize, headSize, scale, softcap, head, ops))).join();
    }

    private void decodePagedAttentionHead(TensorRef output, TensorRef query, TensorRef[] keyPages,
            TensorRef[] valuePages, int visibleRows, int headGroupSize, int headSize, float scale,
            Float softcap, int head, java.util.List<TensorOps> providers) {
        int kvOffset = (head / headGroupSize) * headSize;
        int queryOffset = head * headSize;
        try (TensorRef scores = lighter.allocate(io.teknek.deliverance.DType.F32,
                io.teknek.deliverance.tensor.TensorShape.of(1, visibleRows))) {
            int globalRow = 0;
            for (int pageIndex = 0; pageIndex < keyPages.length; pageIndex++) {
                if (globalRow >= visibleRows) break;
                TensorRef keyPage = keyPages[pageIndex];
                int rows = pageRows(keyPage, pageIndex, keyPages.length, globalRow, visibleRows);
                if (rows <= 0) continue;
                batchDotProduct(providers, new BatchDotProduct().result(scores).a(query).b(keyPage)
                        .aColumnOffset(queryOffset).bColumnOffset(kvOffset).columnLength(headSize)
                        .resultRowOffset(globalRow).bRowOffset(0).rowChunkSize(rows));
                globalRow += rows;
            }
            scaledSoftMaxBody(new ScaledSoftMax(scale).target(scores).offsetAndLength(0, visibleRows)
                    .softcap(softcap), Map.of());
            globalRow = 0;
            for (int pageIndex = 0; pageIndex < valuePages.length; pageIndex++) {
                if (globalRow >= visibleRows) break;
                TensorRef valuePage = valuePages[pageIndex];
                int rows = pageRows(valuePage, pageIndex, valuePages.length, globalRow, visibleRows);
                if (rows <= 0) continue;
                saxpy(providers, scores, valuePage, output, kvOffset, queryOffset, headSize,
                        globalRow, 0, rows);
                globalRow += rows;
            }
        }
    }

    private void batchDotProduct(java.util.List<TensorOps> providers, BatchDotProduct operation) {
        for (TensorOps provider : providers) {
            if (provider.batchDotProduct(operation).isRight()) return;
        }
        throw new IllegalStateException("No tensor provider supports paged attention QK");
    }

    private void saxpy(java.util.List<TensorOps> providers, TensorRef alpha, TensorRef x, TensorRef y,
            int xOffset, int yOffset, int length, int alphaOffset, int xRowOffset, int batchSize) {
        for (TensorOps provider : providers) {
            if (provider.saxpy(alpha, x, y, xOffset, yOffset, length, alphaOffset, xRowOffset, batchSize).isRight()) {
                return;
            }
        }
        throw new IllegalStateException("No tensor provider supports paged attention value accumulation");
    }

    private int pageRows(TensorRef page, int pageIndex, int pageCount, int globalRow, int visibleRows) {
        if (pageIndex == pageCount - 1) {
            return Math.min(page.shape().first(), visibleRows - globalRow);
        }
        int prefixRows = visibleRows - 1;
        return Math.min(page.shape().first(), prefixRows - globalRow);
    }

    public void softMax(ScaledSoftMax operation) {
        scaledSoftMax(operation);
    }

    public void scaledSoftMax(ScaledSoftMax operation) {
        scaledSoftMax(operation, Map.of());
    }

    public void scaledSoftMax(ScaledSoftMax operation, Map<String, String> tags) {
        validateScaledSoftMax(operation);
        run("tensor2.composite.scaled_softmax", operation.getTarget(), operation.getLength(), tags,
                () -> scaledSoftMaxBody(operation, tags));
    }

    private void scaledSoftMaxBody(ScaledSoftMax operation, Map<String, String> tags) {
        TensorRef target = operation.getTarget();
        int offset = operation.getOffset();
        int length = operation.getLength();
        int limit = offset + length;
        float maxVal = transformForAttentionSoftmax(target.underlying().get(0, offset), operation.getScale(),
                operation.getSoftcap());
        for (int i = offset + 1; i < limit; i++) {
            float value = transformForAttentionSoftmax(target.underlying().get(0, i), operation.getScale(),
                    operation.getSoftcap());
            if (value > maxVal) maxVal = value;
        }
        float sum = 0.0f;
        for (int i = offset; i < limit; i++) {
            float value = transformForAttentionSoftmax(target.underlying().get(0, i), operation.getScale(),
                    operation.getSoftcap());
            target.underlying().set((float) FastMath.exp(value - maxVal), 0, i);
            sum += target.underlying().get(0, i);
        }
        multiplyInPlace(new MultiplyInPlace(1.0f / sum).target(target).offsetAndLength(offset, length), tags);
    }

    /** Applies configured RoPE to the contiguous query/key heads in a TensorRef. */
    public void rotaryEmbedding(TensorRef target, int heads, int headSize, int startPosition,
            float[][] ropeFrequencies) {
        Preconditions.checkArgument(target != null, "RoPE target must be set");
        Preconditions.checkArgument(target.dims() == 2, "RoPE target must be 2D");
        Preconditions.checkArgument(heads >= 0 && headSize > 0 && (headSize & 1) == 0,
                "RoPE head shape is invalid");
        Preconditions.checkArgument(startPosition >= 0, "RoPE start position must be non-negative");
        int headPiece = headSize / 2;
        int batchSize = target.shape().first();
        Preconditions.checkArgument(target.shape().last() >= heads * headSize,
                "RoPE target does not contain all heads");
        Preconditions.checkArgument(ropeFrequencies != null
                        && ropeFrequencies.length >= (startPosition + batchSize) * headPiece,
                "RoPE frequencies do not cover the target positions");
        for (int row = 0; row < batchSize; row++) {
            int positionOffset = (startPosition + row) * headPiece;
            for (int head = 0; head < heads; head++) {
                int offset = head * headSize;
                for (int dimension = 0; dimension < headPiece; dimension++) {
                    float x0 = target.get(row, offset + dimension);
                    float x1 = target.get(row, offset + dimension + headPiece);
                    float[] frequency = ropeFrequencies[positionOffset + dimension];
                    float cosine = frequency[0];
                    float sine = frequency[1];
                    target.set(x0 * cosine - x1 * sine, row, offset + dimension);
                    target.set(x0 * sine + x1 * cosine, row, offset + dimension + headPiece);
                }
            }
        }
    }

    /** Applies F32 RMSNorm using vectorized row reductions and scaling. */
    public void rmsNorm(TensorRef output, TensorRef input, TensorRef weights, float epsilon,
            float weightAdjustment) {
        Preconditions.checkArgument(output.dType() == io.teknek.deliverance.DType.F32
                        && input.dType() == io.teknek.deliverance.DType.F32
                        && weights.dType() == io.teknek.deliverance.DType.F32,
                "F32 RMSNorm requires F32 tensors");
        Preconditions.checkArgument(output.shape().equals(input.shape()), "RMSNorm output shape must match input");
        Preconditions.checkArgument(weights.dims() == 2 && weights.shape().first() == 1
                        && weights.shape().last() >= input.shape().last(),
                "RMSNorm weights must be [1, inputWidth]");
        int rows = input.shape().first();
        int columns = input.shape().last();
        MemorySegment inputMemory = input.memorySegment();
        MemorySegment outputMemory = output.memorySegment();
        MemorySegment weightMemory = weights.memorySegment();
        int upper = F32_SPECIES.loopBound(columns);
        for (int row = 0; row < rows; row++) {
            long inputBase = input.memorySegmentOffset(input.shape().getOffset(row, 0));
            long outputBase = output.memorySegmentOffset(output.shape().getOffset(row, 0));
            long weightBase = weights.memorySegmentOffset(weights.shape().getOffset(0, 0));
            FloatVector sum = FloatVector.zero(F32_SPECIES);
            for (int column = 0; column < upper; column += F32_SPECIES.length()) {
                FloatVector values = FloatVector.fromMemorySegment(F32_SPECIES, inputMemory,
                        inputBase + (long) column * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
                sum = values.fma(values, sum);
            }
            float sumSquares = sum.reduceLanes(VectorOperators.ADD);
            for (int column = upper; column < columns; column++) {
                float value = inputMemory.get(java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED,
                        inputBase + (long) column * Float.BYTES);
                sumSquares += value * value;
            }
            float inverseRms = (float) (1.0 / StrictMath.sqrt(sumSquares / columns + epsilon));
            FloatVector inverse = FloatVector.broadcast(F32_SPECIES, inverseRms);
            FloatVector adjustment = FloatVector.broadcast(F32_SPECIES, weightAdjustment);
            for (int column = 0; column < upper; column += F32_SPECIES.length()) {
                long inputAddress = inputBase + (long) column * Float.BYTES;
                FloatVector values = FloatVector.fromMemorySegment(F32_SPECIES, inputMemory, inputAddress,
                        ByteOrder.LITTLE_ENDIAN);
                FloatVector scales = FloatVector.fromMemorySegment(F32_SPECIES, weightMemory,
                        weightBase + (long) column * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
                values.mul(inverse).mul(scales.add(adjustment)).intoMemorySegment(outputMemory,
                        outputBase + (long) column * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
            }
            if (upper < columns) {
                VectorMask<Float> mask = F32_SPECIES.indexInRange(upper, columns);
                long inputAddress = inputBase + (long) upper * Float.BYTES;
                FloatVector values = FloatVector.fromMemorySegment(F32_SPECIES, inputMemory, inputAddress,
                        ByteOrder.LITTLE_ENDIAN, mask);
                FloatVector scales = FloatVector.fromMemorySegment(F32_SPECIES, weightMemory,
                        weightBase + (long) upper * Float.BYTES, ByteOrder.LITTLE_ENDIAN, mask);
                values.mul(inverse).mul(scales.add(adjustment)).intoMemorySegment(outputMemory,
                        outputBase + (long) upper * Float.BYTES, ByteOrder.LITTLE_ENDIAN, mask);
            }
        }
    }

    /** Applies independent in-place RMSNorm groups across each row, matching Q/K head RMSNorm semantics. */
    public void groupedRmsNormInPlace(TensorRef target, int groups, int groupSize, float epsilon,
            TensorRef weights) {
        validateGroupedRmsNorm(target, groups, groupSize, weights);
        if (target.dType() == io.teknek.deliverance.DType.F32
                && (weights == null || weights.dType() == io.teknek.deliverance.DType.F32)) {
            groupedRmsNormF32InPlace(target, groups, groupSize, epsilon, weights);
            return;
        }
        groupedRmsNormScalarInPlace(target, groups, groupSize, epsilon, weights);
    }

    private void groupedRmsNormF32InPlace(TensorRef target, int groups, int groupSize, float epsilon,
            TensorRef weights) {
        MemorySegment targetMemory = target.memorySegment();
        MemorySegment weightMemory = weights == null ? null : weights.memorySegment();
        int rows = target.shape().first();
        int upper = F32_SPECIES.loopBound(groupSize);
        for (int row = 0; row < rows; row++) {
            for (int group = 0; group < groups; group++) {
                int offset = group * groupSize;
                long base = target.memorySegmentOffset(target.shape().getOffset(row, offset));
                FloatVector sum = FloatVector.zero(F32_SPECIES);
                for (int i = 0; i < upper; i += F32_SPECIES.length()) {
                    FloatVector values = FloatVector.fromMemorySegment(F32_SPECIES, targetMemory,
                            base + (long) i * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
                    sum = values.fma(values, sum);
                }
                double sumSquares = sum.reduceLanes(VectorOperators.ADD);
                for (int i = upper; i < groupSize; i++) {
                    float value = targetMemory.get(java.lang.foreign.ValueLayout.JAVA_FLOAT,
                            base + (long) i * Float.BYTES);
                    sumSquares += value * value;
                }
                float invRms = (float) (1.0 / StrictMath.sqrt((sumSquares / groupSize) + epsilon));
                FloatVector inv = FloatVector.broadcast(F32_SPECIES, invRms);
                for (int i = 0; i < upper; i += F32_SPECIES.length()) {
                    long address = base + (long) i * Float.BYTES;
                    FloatVector values = FloatVector.fromMemorySegment(F32_SPECIES, targetMemory, address,
                            ByteOrder.LITTLE_ENDIAN).mul(inv);
                    if (weights != null) {
                        values = values.mul(FloatVector.fromMemorySegment(F32_SPECIES, weightMemory,
                                weights.memorySegmentOffset(weights.shape().getOffset(0, i)), ByteOrder.LITTLE_ENDIAN));
                    }
                    values.intoMemorySegment(targetMemory, address, ByteOrder.LITTLE_ENDIAN);
                }
                for (int i = upper; i < groupSize; i++) {
                    long address = base + (long) i * Float.BYTES;
                    float value = targetMemory.get(java.lang.foreign.ValueLayout.JAVA_FLOAT, address) * invRms;
                    if (weights != null) {
                        value *= weights.get(0, i);
                    }
                    targetMemory.set(java.lang.foreign.ValueLayout.JAVA_FLOAT, address, value);
                }
            }
        }
    }

    private void groupedRmsNormScalarInPlace(TensorRef target, int groups, int groupSize, float epsilon,
            TensorRef weights) {
        int rows = target.shape().first();
        for (int row = 0; row < rows; row++) {
            for (int group = 0; group < groups; group++) {
                int offset = group * groupSize;
                double sumSquares = 0.0;
                for (int i = 0; i < groupSize; i++) {
                    float value = target.get(row, offset + i);
                    sumSquares += value * value;
                }
                double invRms = 1.0 / StrictMath.sqrt((sumSquares / groupSize) + epsilon);
                for (int i = 0; i < groupSize; i++) {
                    float scaled = (float) (target.get(row, offset + i) * invRms);
                    if (weights != null) {
                        scaled *= weights.get(0, i);
                    }
                    target.set(scaled, row, offset + i);
                }
            }
        }
    }

    private static void validateGroupedRmsNorm(TensorRef target, int groups, int groupSize, TensorRef weights) {
        Preconditions.checkArgument(target != null, "RMSNorm target must be set");
        Preconditions.checkArgument(target.dims() == 2, "RMSNorm target must be 2D");
        Preconditions.checkArgument(groups >= 0 && groupSize > 0, "RMSNorm group shape is invalid");
        Preconditions.checkArgument(target.shape().last() >= groups * groupSize,
                "RMSNorm target does not contain all groups");
        if (weights != null) {
            Preconditions.checkArgument(weights.dims() == 2, "RMSNorm weights must be [1, groupSize]");
            Preconditions.checkArgument(weights.shape().first() == 1, "RMSNorm weights must have one row");
            Preconditions.checkArgument(weights.shape().last() >= groupSize,
                    "RMSNorm weights must cover groupSize columns");
        }
    }

    /** Applies vectorized SiLU to an F32 tensor in place. */
    public void silu(TensorRef target) {
        Preconditions.checkArgument(target.dType() == io.teknek.deliverance.DType.F32,
                "F32 SiLU requires an F32 tensor");
        MemorySegment memory = target.memorySegment();
        int rows = target.shape().first();
        int columns = target.shape().last();
        int upper = F32_SPECIES.loopBound(columns);
        for (int row = 0; row < rows; row++) {
            long base = target.memorySegmentOffset(target.shape().getOffset(row, 0));
            for (int column = 0; column < upper; column += F32_SPECIES.length()) {
                long address = base + (long) column * Float.BYTES;
                FloatVector values = FloatVector.fromMemorySegment(F32_SPECIES, memory, address,
                        ByteOrder.LITTLE_ENDIAN);
                FloatVector result = values.div(FloatVector.broadcast(F32_SPECIES, 1.0f)
                        .add(values.neg().lanewise(VectorOperators.EXP)));
                result.intoMemorySegment(memory, address, ByteOrder.LITTLE_ENDIAN);
            }
            if (upper < columns) {
                VectorMask<Float> mask = F32_SPECIES.indexInRange(upper, columns);
                long address = base + (long) upper * Float.BYTES;
                FloatVector values = FloatVector.fromMemorySegment(F32_SPECIES, memory, address,
                        ByteOrder.LITTLE_ENDIAN, mask);
                FloatVector result = values.div(FloatVector.broadcast(F32_SPECIES, 1.0f)
                        .add(values.neg().lanewise(VectorOperators.EXP)));
                result.intoMemorySegment(memory, address, ByteOrder.LITTLE_ENDIAN, mask);
            }
        }
    }

    /** Applies legacy residual semantics: {@code target = multiplier * target + residual}. */
    public void addResidual(TensorRef target, TensorRef residual, float multiplier) {
        Preconditions.checkArgument(target.dType() == io.teknek.deliverance.DType.F32
                        && residual.dType() == io.teknek.deliverance.DType.F32,
                "F32 residuals require F32 tensors");
        Preconditions.checkArgument(target.shape().equals(residual.shape()), "Residual shapes must match");
        MemorySegment targetMemory = target.memorySegment();
        MemorySegment residualMemory = residual.memorySegment();
        int rows = target.shape().first();
        int columns = target.shape().last();
        int upper = F32_SPECIES.loopBound(columns);
        FloatVector factor = FloatVector.broadcast(F32_SPECIES, multiplier);
        for (int row = 0; row < rows; row++) {
            long targetBase = target.memorySegmentOffset(target.shape().getOffset(row, 0));
            long residualBase = residual.memorySegmentOffset(residual.shape().getOffset(row, 0));
            for (int column = 0; column < upper; column += F32_SPECIES.length()) {
                long targetAddress = targetBase + (long) column * Float.BYTES;
                long residualAddress = residualBase + (long) column * Float.BYTES;
                FloatVector values = FloatVector.fromMemorySegment(F32_SPECIES, targetMemory, targetAddress,
                        ByteOrder.LITTLE_ENDIAN);
                FloatVector additions = FloatVector.fromMemorySegment(F32_SPECIES, residualMemory, residualAddress,
                        ByteOrder.LITTLE_ENDIAN);
                values.mul(factor).add(additions).intoMemorySegment(targetMemory, targetAddress,
                        ByteOrder.LITTLE_ENDIAN);
            }
            if (upper < columns) {
                VectorMask<Float> mask = F32_SPECIES.indexInRange(upper, columns);
                long targetAddress = targetBase + (long) upper * Float.BYTES;
                long residualAddress = residualBase + (long) upper * Float.BYTES;
                FloatVector values = FloatVector.fromMemorySegment(F32_SPECIES, targetMemory, targetAddress,
                        ByteOrder.LITTLE_ENDIAN, mask);
                FloatVector additions = FloatVector.fromMemorySegment(F32_SPECIES, residualMemory, residualAddress,
                        ByteOrder.LITTLE_ENDIAN, mask);
                values.mul(factor).add(additions).intoMemorySegment(targetMemory, targetAddress,
                        ByteOrder.LITTLE_ENDIAN, mask);
            }
        }
    }

    private void validateMultiplyInPlace(MultiplyInPlace operation) {
        TensorRef target = operation.getTarget();
        Preconditions.checkArgument(target != null, "Target tensor must be set");
        Preconditions.checkArgument(operation.getOffset() >= 0 && operation.getLength() >= 0
                && operation.getOffset() + operation.getLength() <= target.shape().last(),
                "Multiply window out of bounds");
    }

    private void validateScaledSoftMax(ScaledSoftMax operation) {
        TensorRef target = operation.getTarget();
        Preconditions.checkArgument(target != null, "Target tensor must be set");
        Preconditions.checkArgument(target.shape().first() == 1, "scaledSoftMax requires one row");
        Preconditions.checkArgument(operation.getOffset() >= 0 && operation.getLength() > 0
                && operation.getOffset() + operation.getLength() <= target.shape().last(),
                "Softmax window out of bounds");
    }

    private void run(String metricName, TensorRef target, int length, Map<String, String> tags, Runnable action) {
        if (metricRegistry == null) {
            action.run();
            return;
        }
        Map<String, String> metricTags = new HashMap<>(tags);
        metricTags.put(TENSOR_TYPE, target.dType().name());
        metricTags.put(LENGTH, String.valueOf(length));
        Timer timer = metricRegistry.timer(new MetricName(metricName + ".time", metricTags));
        long startNanos = System.nanoTime();
        action.run();
        metricRegistry.meter(new MetricName(metricName, metricTags)).mark();
        timer.update(System.nanoTime() - startNanos, TimeUnit.NANOSECONDS);
    }

    private static float transformForAttentionSoftmax(float value, float scale, Float softcap) {
        float scaled = value * scale;
        if (softcap == null) {
            return scaled;
        }
        return (float) FastMath.tanh(scaled / softcap) * softcap;
    }
}
