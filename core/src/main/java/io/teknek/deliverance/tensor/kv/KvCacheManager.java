package io.teknek.deliverance.tensor.kv;

import com.google.common.base.Preconditions;
import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorAllocator;

import java.util.Objects;

/** Opens standalone KV cache v2 sessions. */
public final class KvCacheManager {
    private final int layers;
    private final int contextLength;
    private final int kvLength;
    private final int blockSize;
    private final DType dtype;
    private final TensorAllocator allocator;
    private final MetricRegistry metricRegistry;
    private final boolean trackReadViews;
    private final KvBufferCacheSettings settings;

    public KvCacheManager(int layers, int contextLength, int kvLength, DType dtype,
            KvBufferCacheSettings settings, TensorAllocator allocator, MetricRegistry metricRegistry) {
        this(layers, contextLength, kvLength, dtype, settings, allocator, metricRegistry, false);
    }

    public KvCacheManager(int layers, int contextLength, int kvLength, DType dtype,
            KvBufferCacheSettings settings, TensorAllocator allocator, MetricRegistry metricRegistry, boolean trackReadViews) {
        Preconditions.checkArgument(layers > 0, "layers must be > 0");
        Preconditions.checkArgument(contextLength > 0, "contextLength must be > 0");
        Preconditions.checkArgument(kvLength > 0, "kvLength must be > 0");
        this.layers = layers;
        this.contextLength = contextLength;
        this.kvLength = kvLength;
        this.dtype = Objects.requireNonNull(dtype, "dtype");
        this.settings = Objects.requireNonNull(settings, "settings");
        this.blockSize = this.settings.getBlockSize();
        this.allocator = Objects.requireNonNull(allocator, "allocator");
        this.metricRegistry = Objects.requireNonNull(metricRegistry, "metricRegistry");
        this.trackReadViews = trackReadViews;
    }

    public KvCacheSession openSession() {
        metricRegistry.meter("kvcache.v2.session.open").mark();
        return new KvCacheSession(layers, contextLength, kvLength, blockSize, dtype, allocator, metricRegistry,
                trackReadViews, settings);
    }
}
