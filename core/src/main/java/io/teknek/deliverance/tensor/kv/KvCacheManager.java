package io.teknek.deliverance.tensor.kv;

import com.google.common.base.Preconditions;
import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.operations.TensorOperations;
import io.teknek.deliverance.tensor2.Lighter;

import javax.annotation.Nullable;
import java.util.Objects;

/** Opens standalone KV cache v2 sessions. */
public final class KvCacheManager {
    private final int layers;
    private final int contextLength;
    private int kvLength;
    private final int blockSize;
    private final DType dtype;
    private final TensorAllocator allocator;
    private final MetricRegistry metricRegistry;
    private final boolean trackReadViews;
    private final KvBufferCacheSettings settings;
    private final TensorOperations conversionOperations;
    private final Lighter lighter;
    private volatile int[] layerKvLengths;

    public KvCacheManager(int layers, int contextLength, int kvLength, DType dtype,
            KvBufferCacheSettings settings, TensorAllocator allocator, MetricRegistry metricRegistry) {
        this(layers, contextLength, kvLength, dtype, settings, allocator, metricRegistry, false, null);
    }

    public KvCacheManager(int layers, int contextLength, int kvLength, DType dtype,
            KvBufferCacheSettings settings, TensorAllocator allocator, MetricRegistry metricRegistry, boolean trackReadViews) {
        this(layers, contextLength, kvLength, dtype, settings, allocator, metricRegistry, trackReadViews, null);
    }

    public KvCacheManager(int layers, int contextLength, int kvLength, DType dtype,
            KvBufferCacheSettings settings, TensorAllocator allocator, MetricRegistry metricRegistry,
            boolean trackReadViews, @Nullable TensorOperations conversionOperations) {
        this(layers, contextLength, kvLength, dtype, settings, allocator, metricRegistry, trackReadViews,
                conversionOperations, new Lighter(metricRegistry));
    }

    public KvCacheManager(int layers, int contextLength, int kvLength, DType dtype,
            KvBufferCacheSettings settings, TensorAllocator allocator, MetricRegistry metricRegistry,
            boolean trackReadViews, @Nullable TensorOperations conversionOperations, Lighter lighter) {
        Preconditions.checkArgument(layers > 0, "layers must be > 0");
        Preconditions.checkArgument(contextLength > 0, "contextLength must be > 0");
        Preconditions.checkArgument(kvLength > 0, "kvLength must be > 0");
        this.layers = layers;
        this.contextLength = contextLength;
        this.kvLength = kvLength;
        this.dtype = Objects.requireNonNull(dtype, "dtype");
        this.settings = Objects.requireNonNull(settings, "settings");
        this.blockSize = this.settings.getBlockSize();
        // The TensorRef-backed path allocates through Lighter and intentionally has no V1 allocator.
        this.allocator = allocator;
        this.metricRegistry = Objects.requireNonNull(metricRegistry, "metricRegistry");
        this.trackReadViews = trackReadViews;
        this.conversionOperations = conversionOperations;
        this.lighter = Objects.requireNonNull(lighter, "lighter");
    }

    public KvCacheManager(int layers, int contextLength, int kvLength, DType dtype,
            KvBufferCacheSettings settings, MetricRegistry metricRegistry, Lighter lighter) {
        this(layers, contextLength, kvLength, dtype, settings, null, metricRegistry, false, null, lighter);
    }

    public KvCacheManager(int layers, int contextLength, int kvLength, DType dtype,
            KvBufferCacheSettings settings, MetricRegistry metricRegistry, boolean trackReadViews, Lighter lighter) {
        this(layers, contextLength, kvLength, dtype, settings, null, metricRegistry, trackReadViews, null, lighter);
    }

    public void setLayerKvLengths(int[] lengths) {
        Preconditions.checkArgument(lengths != null && lengths.length == layers,
                "layer KV width count must equal layer count");
        int[] copy = lengths.clone();
        int maximum = 0;
        for (int length : copy) {
            Preconditions.checkArgument(length > 0, "layer KV widths must be positive");
            maximum = Math.max(maximum, length);
        }
        Preconditions.checkState(layerKvLengths == null, "layer KV widths already configured");
        layerKvLengths = copy;
        kvLength = maximum;
    }

    public int[] layerKvLengths() {
        return layerKvLengths == null ? null : layerKvLengths.clone();
    }

    public KvCacheSession openSession() {
        metricRegistry.meter("kvcache.v2.session.open").mark();
        return allocator == null
                ? new KvCacheSession(layers, contextLength, kvLength, blockSize, dtype, metricRegistry,
                trackReadViews, settings, conversionOperations, lighter, layerKvLengths)
                : new KvCacheSession(layers, contextLength, kvLength, blockSize, dtype, allocator, metricRegistry,
                trackReadViews, settings, conversionOperations, lighter, layerKvLengths);
    }
}
