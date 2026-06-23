package io.teknek.deliverance.tensor;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.safetensors.Config;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.FileTime;
import java.time.Duration;
import java.time.Instant;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class DiskKvBackendTest {

    @TempDir
    Path tempDir;

    @Test
    public void rangeReadCreatesDistinctFilesForEachContextPage() throws IOException {
        AbstractModel model = mockModel();
        KvBufferCacheSettings settings = new KvBufferCacheSettings(tempDir.toFile())
                .withDiskPageSweeperEnabled(false)
                .withContextRowsPerPageTarget(1);
        KvBufferCache cache = new KvBufferCache(model, settings);
        KvBufferCache.KvBuffer buffer = cache.new KvBuffer("test", 1024);

        try (AbstractTensor laterPage = buffer.getKeyTensorForPosition(0, 3)) {
            laterPage.set(33.0f, 0, 0);
        }

        AbstractTensor[] tensors = buffer.getKeyTensorsUptoPosition(0, 3);
        try {
            assertEquals(4, tensors.length);
            assertTrue(Files.exists(tempDir.resolve("test-L0C0.page")));
            assertTrue(Files.exists(tempDir.resolve("test-L0C1.page")));
            assertTrue(Files.exists(tempDir.resolve("test-L0C2.page")));
            assertTrue(Files.exists(tempDir.resolve("test-L0C3.page")));
            assertNotEquals(tempDir.resolve("test-L0C0.page").toRealPath(),
                    tempDir.resolve("test-L0C1.page").toRealPath());
        } finally {
            for (AbstractTensor tensor : tensors) {
                tensor.close();
            }
            buffer.close();
            cache.close();
        }
    }

    @Test
    public void pagesAreDeletedOnBufferCloseByDefault() {
        AbstractModel model = mockModel();
        KvBufferCacheSettings settings = new KvBufferCacheSettings(tempDir.toFile())
                .withDiskPageSweeperEnabled(false);
        KvBufferCache cache = new KvBufferCache(model, settings);
        Path pagePath = tempDir.resolve("cleanup-L0C0.page");

        KvBufferCache.KvBuffer buffer = cache.new KvBuffer("cleanup", 1024);
        try (AbstractTensor page = buffer.getKeyTensorForPosition(0, 0)) {
            page.set(1.0f, 0, 0);
        }
        assertTrue(Files.exists(pagePath));

        buffer.close();

        assertFalse(Files.exists(pagePath));
        assertEquals(1, model.getMetricRegistry().meter("kvbuffercache.disk.page.create").getCount());
        assertEquals(1, model.getMetricRegistry().meter("kvbuffercache.disk.page.delete").getCount());
        assertEquals(0, model.getMetricRegistry().counter("kvbuffercache.disk.bytes.live").getCount());
        cache.close();
    }

    @Test
    public void diskBackedKvDoesNotStorePrefixSnapshots() throws IOException {
        AbstractModel model = mockModel();
        KvBufferCacheSettings settings = new KvBufferCacheSettings(tempDir.toFile())
                .withDiskPageSweeperEnabled(false);
        KvBufferCache cache = new KvBufferCache(model, settings);
        KvBufferCache.KvBuffer buffer = cache.new KvBuffer("prefix", 1024);

        try (AbstractTensor key = buffer.getKeyTensorForPosition(0, 0);
             AbstractTensor value = buffer.getValTensorForPosition(0, 0)) {
            key.set(1.0f, 0, 0);
            value.set(2.0f, 0, 0);
        }

        cache.storePrefix(new int[]{1, 2, 3, 4, 5, 6, 7, 8}, buffer, java.util.Optional.empty());

        assertEquals(0, cache.prefixCache.size());
        assertEquals(1, model.getMetricRegistry().meter("kvbuffercache.prefix.disk.skip").getCount());
        assertEquals(1, countPageFiles());

        buffer.close();
        cache.close();

        assertEquals(0, countPageFiles());
    }

    @Test
    public void pagesCanBeRetainedForInspection() {
        AbstractModel model = mockModel();
        KvBufferCacheSettings settings = new KvBufferCacheSettings(tempDir.toFile())
                .withDeleteDiskPagesOnClose(false)
                .withDiskPageSweeperEnabled(false);
        KvBufferCache cache = new KvBufferCache(model, settings);
        Path pagePath = tempDir.resolve("retained-L0C0.page");

        try (KvBufferCache.KvBuffer buffer = cache.new KvBuffer("retained", 1024);
             AbstractTensor page = buffer.getKeyTensorForPosition(0, 0)) {
            page.set(2.0f, 0, 0);
        }

        assertTrue(Files.exists(pagePath));
        assertEquals(1, model.getMetricRegistry().meter("kvbuffercache.disk.page.create").getCount());
        assertEquals(0, model.getMetricRegistry().meter("kvbuffercache.disk.page.delete").getCount());
        assertTrue(model.getMetricRegistry().counter("kvbuffercache.disk.bytes.live").getCount() > 0);
        cache.close();
    }

    @Test
    public void sweepDeletesOldClosedPages() throws IOException {
        AbstractModel model = mockModel();
        KvBufferCacheSettings settings = new KvBufferCacheSettings(tempDir.toFile())
                .withDeleteDiskPagesOnClose(false)
                .withDiskPageSweeperEnabled(false)
                .withDiskPageMaxAge(Duration.ofMinutes(1));
        KvBufferCache cache = new KvBufferCache(model, settings);
        Path pagePath = tempDir.resolve("old-L0C0.page");

        try (KvBufferCache.KvBuffer buffer = cache.new KvBuffer("old", 1024);
             AbstractTensor page = buffer.getKeyTensorForPosition(0, 0)) {
            page.set(3.0f, 0, 0);
        }
        Files.setLastModifiedTime(pagePath, FileTime.from(Instant.now().minus(Duration.ofHours(2))));

        cache.runDiskPageSweep();

        assertFalse(Files.exists(pagePath));
        assertEquals(1, model.getMetricRegistry().meter("kvbuffercache.disk.sweeper.run").getCount());
        assertEquals(1, model.getMetricRegistry().meter("kvbuffercache.disk.sweeper.page.delete").getCount());
        assertEquals(0, model.getMetricRegistry().counter("kvbuffercache.disk.bytes.live").getCount());
        cache.close();
    }

    @Test
    public void sweepSkipsActivePagesEvenWhenOld() throws IOException {
        AbstractModel model = mockModel();
        KvBufferCacheSettings settings = new KvBufferCacheSettings(tempDir.toFile())
                .withDeleteDiskPagesOnClose(false)
                .withDiskPageSweeperEnabled(false)
                .withDiskPageMaxAge(Duration.ofMinutes(1));
        KvBufferCache cache = new KvBufferCache(model, settings);
        Path pagePath = tempDir.resolve("active-L0C0.page");
        KvBufferCache.KvBuffer buffer = cache.new KvBuffer("active", 1024);

        try (AbstractTensor page = buffer.getKeyTensorForPosition(0, 0)) {
            page.set(4.0f, 0, 0);
        }
        Files.setLastModifiedTime(pagePath, FileTime.from(Instant.now().minus(Duration.ofHours(2))));

        cache.runDiskPageSweep();

        assertTrue(Files.exists(pagePath));
        assertEquals(1, model.getMetricRegistry().meter("kvbuffercache.disk.sweeper.page.skip.active").getCount());

        buffer.close();
        cache.runDiskPageSweep();

        assertFalse(Files.exists(pagePath));
        cache.close();
    }

    private static AbstractModel mockModel() {
        Config config = new Config(128, 64, 128, 4,
                4, 2, 1e-5f, 1000, 0,
                List.of(1), ActivationFunction.Type.GELU_PYTORCH_TANH, null, null);
        AbstractModel model = mock(AbstractModel.class);
        when(model.getConfig()).thenReturn(config);
        when(model.getLocalKvLength()).thenReturn(config.kvLength);
        when(model.getWorkingDType()).thenReturn(DType.F32);
        when(model.getTensorAllocator()).thenReturn(new ArrayQueueTensorAllocator(new MetricRegistry()));
        when(model.getMetricRegistry()).thenReturn(new MetricRegistry());
        return model;
    }

    private long countPageFiles() throws IOException {
        try (var files = Files.list(tempDir)) {
            return files.filter(path -> path.getFileName().toString().endsWith(".page")).count();
        }
    }
}
