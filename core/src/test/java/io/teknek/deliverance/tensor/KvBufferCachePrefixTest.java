package io.teknek.deliverance.tensor;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.safetensors.DistributedContext;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class KvBufferCachePrefixTest {

    private AbstractModel mockModel() {
        Config config = new Config(128, 64, 128, 4,
                4, 2, 1e-5f, 1000, 0,
                List.of(1), ActivationFunction.Type.GELU_PYTORCH_TANH, null, null);
        DistributedContext dctx = DistributedContext.builder(config).build();
        config.setDistributedContext(dctx);
        AbstractModel model = mock(AbstractModel.class);
        when(model.getConfig()).thenReturn(config);
        when(model.getWorkingDType()).thenReturn(DType.F32);
        when(model.getTensorAllocator()).thenReturn(new ArrayQueueTensorAllocator(new MetricRegistry()));
        when(model.getMetricRegistry()).thenReturn(new MetricRegistry());
        return model;
    }

    @Test
    public void testExactPrefixHit() {
        KvBufferCacheSettings settings = new KvBufferCacheSettings(true).withMaxEntries(512);
        KvBufferCache cache = new KvBufferCache(mockModel(), settings);
        int[] tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        KvBufferCache.KvBuffer buf = cache.getEphemeralKvBuffer();
        cache.storePrefix(tokens, buf, Optional.empty());

        {
            int[] tokens2 = {1, 2, 3};
            KvBufferCache.KvBuffer buf2 = cache.getEphemeralKvBuffer();
            cache.storePrefix(tokens2, buf2, Optional.empty());
        }

        KvBufferCache.PrefixEntry e = cache.lookupPrefix(tokens, Optional.empty());
        assertEquals(tokens.length - 1, e.length());
    }

    @Test
    public void testDisabled() {
        KvBufferCacheSettings settings = new KvBufferCacheSettings(true).withMaxEntries(0);
        KvBufferCache cache = new KvBufferCache(mockModel(), settings);
        int[] tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        KvBufferCache.KvBuffer buf = cache.getEphemeralKvBuffer();
        cache.storePrefix(tokens, buf, Optional.empty());
        assertNull(cache.lookupPrefix(tokens, Optional.empty()));
    }

    @Test
    public void testSaltyDisabled() {
        KvBufferCacheSettings settings = new KvBufferCacheSettings(true).withMaxEntries(10);
        KvBufferCache cache = new KvBufferCache(mockModel(), settings);
        int[] tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        KvBufferCache.KvBuffer buf = cache.getEphemeralKvBuffer();
        cache.storePrefix(tokens, buf, Optional.of("stillsalty"));

        //second
        KvBufferCache.KvBuffer buf2 = cache.getEphemeralKvBuffer();
        cache.storePrefix(tokens, buf2, Optional.of("yolo"));

        assertNotSame(cache.lookupPrefix(tokens, Optional.of("yolo")),
                cache.lookupPrefix(tokens, Optional.of("stillsalty")));
    }

    @Test
    public void keyRowsRoundTripThroughPagesInOrder() {
        KvBufferCacheSettings settings = new KvBufferCacheSettings(true).withBlockSize(1);
        KvBufferCache cache = new KvBufferCache(mockModel(), settings);
        KvBufferCache.KvBuffer buffer = cache.new KvBuffer("test", 1024);

        for (int pos = 0; pos < 5; pos++) {
            try (AbstractTensor row = buffer.getKeyTensorForPosition(0, pos)) {
                row.set(pos * 10 + 1, 0, 0);
                row.set(pos * 10 + 2, 0, 1);
                row.set(pos * 10 + 3, 0, 2);
                row.set(pos * 10 + 4, 0, 3);
            }
        }


        try (AbstractTensor packed = new FloatBufferTensor(5, 4)) {
            AbstractTensor[] pages = buffer.getKeyTensorsUptoPosition(0, 4);
            try {
                assertEquals(5, fillVisibleRows(packed, pages, 4, 0, 4));
                assertEquals(1.0f, packed.get(0, 0));
                assertEquals(2.0f, packed.get(0, 1));
                assertEquals(3.0f, packed.get(0, 2));
                assertEquals(4.0f, packed.get(0, 3));

                assertEquals(11.0f, packed.get(1, 0));
                assertEquals(12.0f, packed.get(1, 1));
                assertEquals(13.0f, packed.get(1, 2));
                assertEquals(14.0f, packed.get(1, 3));

                assertEquals(21.0f, packed.get(2, 0));
                assertEquals(22.0f, packed.get(2, 1));
                assertEquals(23.0f, packed.get(2, 2));
                assertEquals(24.0f, packed.get(2, 3));

                assertEquals(31.0f, packed.get(3, 0));
                assertEquals(32.0f, packed.get(3, 1));
                assertEquals(33.0f, packed.get(3, 2));
                assertEquals(34.0f, packed.get(3, 3));

                assertEquals(41.0f, packed.get(4, 0));
                assertEquals(42.0f, packed.get(4, 1));
                assertEquals(43.0f, packed.get(4, 2));
                assertEquals(44.0f, packed.get(4, 3));
            } finally {
                for (AbstractTensor page : pages) {
                    page.close();
                }
                buffer.close();
            }
        }
    }

    private static int fillVisibleRows(AbstractTensor packed, AbstractTensor[] pages, int position, int windowStart, int rowWidth) {
        int packedRow = 0;
        int globalOffset = 0;
        for (AbstractTensor page : pages) {
            int pageRows = Math.min(page.shape().first(), (position + 1) - globalOffset);
            int overlapStart = Math.max(windowStart, globalOffset);
            int overlapEnd = Math.min(position + 1, globalOffset + pageRows);
            if (overlapStart < overlapEnd) {
                int rowOffset = overlapStart - globalOffset;
                int size = overlapEnd - overlapStart;
                packed.copyFrom(page, page.getOffset(rowOffset, 0), packed.getOffset(packedRow, 0), size * rowWidth);
                packedRow += size;
            }
            globalOffset += page.shape().first();
        }
        return packedRow;
    }

}
