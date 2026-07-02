package io.teknek.deliverance.tensor;

import com.google.common.base.Preconditions;
import io.teknek.deliverance.DType;

import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.tensor.impl.BFloat16BufferTensor;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.ShortBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * A cache for key-value buffers used in the model.
 *
 * <p>The disk-backed mode is an active-page storage backend for live {@link KvBuffer} instances. It is not a durable
 * prefix cache: disk page names are session/page coordinates, not token-prefix keys, and pages are deleted when their
 * owning buffer closes unless explicitly retained for inspection. A background sweeper can also remove stale closed or
 * orphaned page files from the working directory.</p>
 *
 * <p>This cache stores complete block-aligned prompt prefixes. For a prompt with 9 runtime tokens and a block
 * size of 8, only the first 8 tokens are eligible for reuse; the suffix token must still be run through the model
 * at position 8. The cache key is the token prefix plus an optional salt supplied by generation parameters.</p>
 *
 * <p>This class guarantees only the mechanical cache behavior: block-aligned lookup, snapshot storage, and copying the
 * stored key/value rows back into another KV buffer. Tests in {@code KvBufferCachePrefixTest} assert that round trip at
 * the tensor-value level.</p>
 *
 * <p>Whether using this cache preserves generated output is a larger model-execution invariant involving
 * {@code AbstractModel.generate}, split prefill, attention, quantization, and the selected tensor provider. See
 * {@code core/PrefixCache.md} for the project-level contract and caveats.</p>
 */
public class KvBufferCache implements Closeable {

    public record CacheKey (Optional<String> salt, List<Integer> prefixTokens){
    }
    public record PrefixEntry(KvBuffer buffer, int length) {

    }

    private static final Logger logger = LoggerFactory.getLogger(KvBufferCache.class);

    private final AbstractModel model;
    private final KvBufferCacheSettings kvBufferCacheSettings;
    private final int blockSize;
    private final Set<Path> activeDiskPages = Collections.synchronizedSet(new HashSet<>());
    private final Map<Path, Long> diskPageBytes = Collections.synchronizedMap(new HashMap<>());
    private final AtomicBoolean closed = new AtomicBoolean(false);
    private ScheduledExecutorService diskSweeperExecutor;

    public final Map<CacheKey, PrefixEntry> prefixCache = Collections.synchronizedMap(
            new LinkedHashMap<CacheKey, PrefixEntry>(16, 0.75f, true) {
                public boolean removeEldestEntry(Map.Entry<CacheKey, PrefixEntry> eldest) {
                    boolean evict = size() > kvBufferCacheSettings.getMaxEntries();
                    if (evict && eldest != null && eldest.getValue() != null) {
                        model.getMetricRegistry().meter("kvbuffercache.evict").mark();
                        try {
                            eldest.getValue().buffer.close();
                        } catch (RuntimeException e) {
                            logger.warn("could not close tensor in cache", e);
                        }
                    }
                    return evict;
                }
            });

    public KvBufferCache(AbstractModel model, KvBufferCacheSettings kvBufferCacheSettings) {
        this.model = model;
        this.kvBufferCacheSettings = kvBufferCacheSettings;
        this.blockSize = kvBufferCacheSettings.getBlockSize();
        prepareDiskWorkingDirectory();
        startDiskPageSweeper();
    }

    private void prepareDiskWorkingDirectory() {
        if (kvBufferCacheSettings.isEphemeral()) {
            return;
        }
        File workingDirectory = kvBufferCacheSettings.getWorkingDirectory();
        if (workingDirectory == null) {
            model.getMetricRegistry().meter("kvbuffercache.disk.directory.error").mark();
            throw new IllegalStateException("Disk KV cache requires a workingDirectory");
        }
        try {
            Files.createDirectories(workingDirectory.toPath());
        } catch (IOException e) {
            model.getMetricRegistry().meter("kvbuffercache.disk.directory.error").mark();
            throw new IOError(e);
        }
        if (!workingDirectory.isDirectory()) {
            model.getMetricRegistry().meter("kvbuffercache.disk.directory.error").mark();
            throw new IllegalArgumentException("KV disk workingDirectory must be a directory: " + workingDirectory);
        }
    }

    private void startDiskPageSweeper() {
        if (kvBufferCacheSettings.isEphemeral() || !kvBufferCacheSettings.isDiskPageSweeperEnabled()) {
            return;
        }
        diskSweeperExecutor = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread thread = new Thread(r,
                    "deliverance-kv-disk-sweeper-" + Integer.toHexString(System.identityHashCode(this)));
            thread.setDaemon(true);
            return thread;
        });
        long intervalMillis = kvBufferCacheSettings.getDiskPageSweepInterval().toMillis();
        diskSweeperExecutor.scheduleWithFixedDelay(this::runDiskPageSweepSafely,
                intervalMillis, intervalMillis, TimeUnit.MILLISECONDS);
    }

    private void runDiskPageSweepSafely() {
        try {
            runDiskPageSweep();
        } catch (RuntimeException e) {
            model.getMetricRegistry().meter("kvbuffercache.disk.sweeper.error").mark();
            logger.warn("KV disk page sweeper failed", e);
        }
    }

    /**
     * Removes stale disk-backed KV page files from the active-page working directory.
     *
     * <p>The sweep is age based: only {@code *.page} files older than
     * {@link KvBufferCacheSettings#getDiskPageMaxAge()} are eligible. Pages currently open by this cache instance are
     * skipped even if their file timestamp is old. This is a cleanup safety net for orphaned or intentionally retained
     * active KV page files, not a persistent prefix-cache manifest or token-index cleanup process.</p>
     */
    public void runDiskPageSweep() {
        if (kvBufferCacheSettings.isEphemeral()) {
            return;
        }
        model.getMetricRegistry().meter("kvbuffercache.disk.sweeper.run").mark();
        Path workingDirectory = kvBufferCacheSettings.getWorkingDirectory().toPath();
        Instant deleteBefore = Instant.now().minus(kvBufferCacheSettings.getDiskPageMaxAge());
        try (DirectoryStream<Path> pages = Files.newDirectoryStream(workingDirectory, "*.page")) {
            for (Path page : pages) {
                model.getMetricRegistry().meter("kvbuffercache.disk.sweeper.page.scan").mark();
                Path normalizedPage = page.toAbsolutePath().normalize();
                if (activeDiskPages.contains(normalizedPage)) {
                    model.getMetricRegistry().meter("kvbuffercache.disk.sweeper.page.skip.active").mark();
                    continue;
                }
                Instant lastModified = Files.getLastModifiedTime(page).toInstant();
                if (!lastModified.isBefore(deleteBefore)) {
                    model.getMetricRegistry().meter("kvbuffercache.disk.sweeper.page.skip.young").mark();
                    continue;
                }
                deleteDiskPageFile(normalizedPage, true);
            }
        } catch (IOException e) {
            model.getMetricRegistry().meter("kvbuffercache.disk.sweeper.error").mark();
            throw new IOError(e);
        }
    }

    private Path normalizeDiskPagePath(Path path) {
        return path.toAbsolutePath().normalize();
    }

    private void deleteDiskPageFile(Path path, boolean fromSweeper) throws IOException {
        long fileBytes = 0;
        if (Files.exists(path)) {
            fileBytes = Files.size(path);
        }
        if (Files.deleteIfExists(path)) {
            model.getMetricRegistry().meter("kvbuffercache.disk.page.delete").mark();
            model.getMetricRegistry().counter("kvbuffercache.disk.bytes.deleted").inc(fileBytes);
            Long trackedBytes = diskPageBytes.remove(path);
            if (trackedBytes != null) {
                model.getMetricRegistry().counter("kvbuffercache.disk.bytes.live").dec(trackedBytes);
            }
            if (fromSweeper) {
                model.getMetricRegistry().meter("kvbuffercache.disk.sweeper.page.delete").mark();
            }
        }
    }

    public PrefixEntry lookupPrefix(int[] tokens, Optional<String> salt) {
        PrefixEntry best = null;
        int limit = kvBufferCacheSettings.getMaxPrefixTokensPerPrompt();
        for (int prefixLen : checkpointLengths(Math.min(tokens.length, limit))) {
            PrefixEntry e = prefixCache.get(new CacheKey(salt, prefixTokens(tokens, prefixLen)));
            if (e != null) {
                best = e;
            }
        }
        model.getMetricRegistry().meter("kvbuffercache.lookup").mark();
        if (best != null && best.length >= blockSize && best.length % blockSize == 0) {
            model.getMetricRegistry().meter("kvbuffercache.hits").mark();
            return best;
        }
        model.getMetricRegistry().meter("kvbuffercache.misses").mark();
        return null;
    }

    public void storePrefix(int[] tokens, KvBuffer buffer, Optional<String> salt) {
        if (!kvBufferCacheSettings.isEphemeral()) {
            model.getMetricRegistry().meter("kvbuffercache.prefix.disk.skip").mark();
            return;
        }
        if (kvBufferCacheSettings.getMaxEntries() < 1){
            return;
        }
        int limit = kvBufferCacheSettings.getMaxPrefixTokensPerPrompt();
        for (int prefixLen : checkpointLengths(Math.min(tokens.length, limit))) {
            KvBuffer snapshot = getEphemeralKvBuffer();
            copyPrefix(buffer, snapshot, prefixLen);
            prefixCache.putIfAbsent(new CacheKey(salt, prefixTokens(tokens, prefixLen)), new PrefixEntry(snapshot, prefixLen));
        }
    }

    List<Integer> checkpointLengths(int tokenLength) {
        if (tokenLength < blockSize) {
            return List.of();
        }
        int largest = (tokenLength / blockSize) * blockSize;
        if (largest < blockSize) {
            return List.of();
        }
        if (kvBufferCacheSettings.getPrefixCheckpointPolicy() == KvBufferCacheSettings.PrefixCheckpointPolicy.FIXED_BLOCKS) {
            ArrayList<Integer> fixed = new ArrayList<>();
            for (int prefixLen = blockSize; prefixLen <= largest; prefixLen += blockSize) {
                fixed.add(prefixLen);
            }
            return fixed;
        }
        if (kvBufferCacheSettings.getPrefixCheckpointPolicy() == KvBufferCacheSettings.PrefixCheckpointPolicy.START_AND_END) {
            int max = kvBufferCacheSettings.getMaxPrefixCheckpointsPerPrompt();
            int startCount = (max + 1) / 2;
            int endCount = max - startCount;
            LinkedHashSet<Integer> selected = new LinkedHashSet<>();
            for (int prefixLen = blockSize; prefixLen <= largest && selected.size() < startCount; prefixLen += blockSize) {
                selected.add(prefixLen);
            }
            for (int prefixLen = largest - ((endCount - 1) * blockSize); prefixLen <= largest; prefixLen += blockSize) {
                if (prefixLen >= blockSize) {
                    selected.add(prefixLen);
                }
            }
            return selected.stream().sorted().toList();
        }
        int max = kvBufferCacheSettings.getMaxPrefixCheckpointsPerPrompt();
        LinkedHashSet<Integer> selected = new LinkedHashSet<>();
        for (Integer anchor : kvBufferCacheSettings.getPrefixCheckpointAnchors()) {
            int aligned = (anchor / blockSize) * blockSize;
            if (aligned >= blockSize && aligned <= largest) {
                selected.add(aligned);
            }
            if (selected.size() >= Math.max(0, max - 1)) {
                break;
            }
        }
        selected.add(largest);
        ArrayList<Integer> result = new ArrayList<>(selected);
        result.sort(Integer::compareTo);
        if (result.size() > max) {
            ArrayList<Integer> trimmed = new ArrayList<>(result.subList(0, max - 1));
            trimmed.add(largest);
            return trimmed.stream().distinct().sorted().toList();
        }
        return result;
    }

    private static List<Integer> prefixTokens(int[] tokens, int prefixLen) {
        ArrayList<Integer> prefix = new ArrayList<>(prefixLen);
        for (int i = 0; i < prefixLen; i++) {
            prefix.add(tokens[i]);
        }
        return prefix;
    }

    public void copyPrefix(KvBuffer src, KvBuffer dest, int length) {
        Config c = model.getConfig();
        int layers = c.numberOfLayers;
        for (int layer = 0; layer < layers; layer++) {
            for (int pos = 0; pos < length; pos++) {
                AbstractTensor srcK = src.getKeyTensorForPosition(layer, pos);
                AbstractTensor srcV = src.getValTensorForPosition(layer, pos);

                AbstractTensor dstK = dest.getKeyTensorForPosition(layer, pos);
                AbstractTensor dstV = dest.getValTensorForPosition(layer, pos);
                dstK.copyFrom(srcK, 0, 0, (int) srcK.size());
                dstV.copyFrom(srcV, 0, 0, (int) srcV.size());
                srcK.close();
                srcV.close();
                dstK.close();
                dstV.close();
            }
        }
        dest.setCurrentContextPosition(length);
    }

    public KvBuffer getEphemeralKvBuffer() {
        return new KvBuffer(UUID.randomUUID().toString(), 1 << 20);
    }

    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            if (diskSweeperExecutor != null) {
                diskSweeperExecutor.shutdownNow();
            }
            prefixCache.entrySet().iterator().forEachRemaining(e -> e.getValue().buffer().close());
        }
    }

    class KvPageContext {
        public final int numberOfLayerPages;
        public final int numberOfContextPages;
        private final int layersPerPage;
        private final int contextLengthPerPage;
        private final String session;

        public final TensorShape pageShape;

        public KvPageContext(String session, int numberOfLayerPages, int numberOfContextPages, int layersPerPage, int contextLengthPerPage) {
            this.session = session;
            this.numberOfLayerPages = numberOfLayerPages;
            this.numberOfContextPages = numberOfContextPages;
            this.layersPerPage = layersPerPage;
            this.contextLengthPerPage = contextLengthPerPage;

            if (numberOfLayerPages < 1) {
                throw new IllegalArgumentException("totalPageCount must be >= 1");
            }
            if (numberOfContextPages < 1) {
                throw new IllegalArgumentException("numberOfContextPages must be >= 1");
            }
            if (layersPerPage < 1) {
                throw new IllegalArgumentException("layersPerPage must be >= 1");
            }
            if (contextLengthPerPage < 1) {
                throw new IllegalArgumentException("contextLengthPerPage must be >= 1");
            }

            TensorShape s;
            Config c = model.getConfig();
            int[] rawShape = new int[]{layersPerPage, 2, contextLengthPerPage, model.getLocalKvLength()};
            s = TensorShape.of(rawShape);
            this.pageShape = s;
        }

        int layersPerPage() {
            return layersPerPage;
        }

        int contextLengthPerPage() {
            return contextLengthPerPage;
        }
    }

    /**
     * A Page of a key-value buffer.
     * Rather than allocating one giant buffer for the entire key-value buffer, we allocate slices of the buffer
     * as needed. This allows us to keep the memory usage low, and also allows us to allocate very large contexts.
     */
    class KvBufferPage implements AutoCloseable {
        private final AbstractTensor tensor;

        private final AtomicBoolean closed = new AtomicBoolean(false);
        private final RandomAccessFile raf;
        private final Path diskPath;

        KvBufferPage(KvPageContext pageCtx, String pageId) {
            //this looks more and more like two subclasses vs an if statement
            if (kvBufferCacheSettings.isEphemeral()) {
                this.raf = null;
                this.diskPath = null;
                TensorAllocator tc = kvBufferCacheSettings.getDedicatedCache() == null ?
                        model.getTensorAllocator() : kvBufferCacheSettings.getDedicatedCache();
                this.tensor = tc.get(model.getWorkingDType(), pageCtx.pageShape);
            } else {
                Path path = normalizeDiskPagePath(Paths.get(
                        kvBufferCacheSettings.getWorkingDirectory().toString(),
                        pageCtx.session.toString() + "-" + pageId + ".page"
                ));
                long bytes = pageCtx.pageShape.size() * model.getWorkingDType().size();
                boolean existed = Files.exists(path);
                try {
                    raf = new RandomAccessFile(path.toFile(), "rw");
                    model.getMetricRegistry().meter("kvbuffercache.disk.page.open").mark();
                    if (!existed) {
                        model.getMetricRegistry().meter("kvbuffercache.disk.page.create").mark();
                        model.getMetricRegistry().counter("kvbuffercache.disk.bytes.allocated").inc(bytes);
                        model.getMetricRegistry().counter("kvbuffercache.disk.bytes.live").inc(bytes);
                        diskPageBytes.put(path, bytes);
                    }
                    logger.debug("Allocating page {} with {} bytes {}", pageId, bytes, raf.length());
                    if (raf.length() != bytes) {
                        long previousLength = raf.length();
                        raf.setLength(bytes);
                        if (existed && bytes > previousLength) {
                            model.getMetricRegistry().counter("kvbuffercache.disk.bytes.allocated")
                                    .inc(bytes - previousLength);
                        }
                    }

                    AbstractTensor<?, ?> t;
                    if (model.getWorkingDType() == DType.F32) {
                        FloatBuffer fb = raf.getChannel()
                                .map(FileChannel.MapMode.READ_WRITE, 0, bytes)
                                .order(ByteOrder.LITTLE_ENDIAN)
                                .asFloatBuffer();

                        t = new FloatBufferTensor(fb, pageCtx.pageShape, true);
                    } else if (model.getWorkingDType() == DType.BF16) {
                        ShortBuffer sb = raf.getChannel()
                                .map(FileChannel.MapMode.READ_WRITE, 0, bytes)
                                .order(ByteOrder.LITTLE_ENDIAN)
                                .asShortBuffer();

                        t = new BFloat16BufferTensor("kvmem", sb, pageCtx.pageShape, true);
                    } else {
                        throw new UnsupportedOperationException("Only F32/BF16 is supported for now");
                    }
                    this.tensor = t;
                    this.diskPath = path;
                    activeDiskPages.add(path);
                } catch (IOException e) {
                    model.getMetricRegistry().meter("kvbuffercache.disk.page.open.error").mark();
                    throw new IOError(e);
                }
            }
        }

        public AbstractTensor getTensor() {
            if (closed.get()) {
                model.getMetricRegistry().meter("kvbuffercache.page.closed.access").mark();
                logger.warn("Attempted to access a closed KV buffer page");
                throw new IllegalStateException("KV buffer page is closed");
            }
            return tensor;
        }

        public boolean isClosed() {
            return closed.get();
        }

        @Override
        public void close() throws IOException {
            if (closed.compareAndSet(false, true)) {
                try {
                    if (raf != null) {
                        raf.close();
                    }
                    tensor.close();
                    if (diskPath != null) {
                        model.getMetricRegistry().meter("kvbuffercache.disk.page.close").mark();
                    }
                } catch (IOException | RuntimeException e) {
                    if (diskPath != null) {
                        model.getMetricRegistry().meter("kvbuffercache.disk.page.close.error").mark();
                    }
                    throw e;
                }
                if (diskPath != null && kvBufferCacheSettings.isDeleteDiskPagesOnClose()) {
                    try {
                        deleteDiskPageFile(diskPath, false);
                    } catch (IOException | RuntimeException e) {
                        model.getMetricRegistry().meter("kvbuffercache.disk.page.delete.error").mark();
                        throw e;
                    } finally {
                        activeDiskPages.remove(diskPath);
                    }
                } else if (diskPath != null) {
                    activeDiskPages.remove(diskPath);
                }
            }
        }
    }

    public class KvBuffer implements AutoCloseable {
        private final String session;
        private final AtomicInteger currentContextPosition = new AtomicInteger(0);
        private final KvBufferPage[][] pages;

        private final KvPageContext pageContext;

        KvBuffer(String session, int maxPageSizeInBytes) {
            this.session = session;
            this.pageContext = computePageSize(maxPageSizeInBytes);
            this.pages = new KvBufferPage[pageContext.numberOfLayerPages][pageContext.numberOfContextPages];
        }

        public int getCurrentContextPosition() {
            return currentContextPosition.get();
        }

        public void setCurrentContextPosition(int position) {
            currentContextPosition.set(position);
        }

        /**
         * Currently this is called by AbstractModel inside the generation loop. The position starts at 0
         * interestingly the 0th token has special properties with respect to increment
         */
        public void incrementContextPosition() {
            currentContextPosition.incrementAndGet();
        }

        /**
         * Chooses the active KV page shape under a maximum page byte budget.
         *
         * <p>Each KV page stores both key and value tensors for some number of layers and some number of context-token
         * rows. The cost of one layer at one context position is:</p>
         *
         * <pre>{@code
         * bytesPerLayerToken = 2 * workingDType.bytes * localKvLength
         * }</pre>
         *
         * <p>The factor of {@code 2} is key plus value. {@code localKvLength} is the per-rank KV width, normally
         * {@code numKeyValueHeads * headDim} divided by tensor-parallel rank count when tensor parallelism is active.</p>
         *
         * <p>Attention reads one layer across many context positions, so this method first tries to honor
         * {@link KvBufferCacheSettings#getContextRowsPerPageTarget()} for context locality. It then packs as many layers
         * as possible into the same page without exceeding {@code maxPageSizeInBytes}. For example, Qwen3-4B with F32 KV
         * has {@code localKvLength=1024}, so one layer-token pair is {@code 8192} bytes. With a 1 MiB page budget and the
         * default 32-row target, the selected layout is {@code 4 layers x 32 context rows}.</p>
         */
        public KvPageContext computePageSize(long maxPageSizeInBytes) {
            Config c = model.getConfig();
            DType workingDType = model.getWorkingDType();
            long s = 2L * workingDType.size() * model.getLocalKvLength(); // Size per layer per context

            Preconditions.checkArgument(maxPageSizeInBytes > s, "maxPageSizeInBytes must be greater than the size of a single layer");

            int N = c.numberOfLayers;
            int C = c.contextLength;

            long maxContextRowsForSingleLayer = Math.max(1, maxPageSizeInBytes / s);
            int optimalContextLengthPerPage = (int) Math.min(C,
                    Math.min(kvBufferCacheSettings.getContextRowsPerPageTarget(), maxContextRowsForSingleLayer));
            int optimalLayersPerPage = (int) Math.max(1,
                    Math.min(N, maxPageSizeInBytes / (optimalContextLengthPerPage * s)));

            // Calculate the number of pages needed
            int numberOfLayerPages = (int) Math.ceil((double) N / optimalLayersPerPage);
            int numberOfContextPages = (int) Math.ceil((double) C / optimalContextLengthPerPage);

            // Calculate the size of each page
            long pageSize = optimalLayersPerPage * optimalContextLengthPerPage * s;

            if (pageSize > maxPageSizeInBytes) {
                throw new IllegalArgumentException(
                        "Calculation error: pageSize > maxPageSizeInBytes: " + pageSize + " > " + maxPageSizeInBytes
                );
            }

            logger.debug(
                    "Optimal page size: {} layers, {} context length, {} bytes, {} layer pages, {} length pages",
                    optimalLayersPerPage,
                    optimalContextLengthPerPage,
                    pageSize,
                    numberOfLayerPages,
                    numberOfContextPages
            );

            return new KvPageContext(session, numberOfLayerPages, numberOfContextPages, optimalLayersPerPage, optimalContextLengthPerPage);
        }

        @Override
        public void close() {
            for (KvBufferPage[] layerPages : pages) {
                if (layerPages != null) {
                    for (KvBufferPage page : layerPages) {
                        if (page != null) {
                            try {
                                page.close();
                            } catch (IOException e) {
                                logger.debug("Error closing page", e);
                            }
                        }
                    }
                }
            }
        }

        public AbstractTensor getKeyTensorForPosition(int layerIndex, int position) {
            return getTensorForPosition(layerIndex, position, 0);
        }

        public AbstractTensor getValTensorForPosition(int layerIndex, int position) {
            return getTensorForPosition(layerIndex, position, 1);
        }

        private AbstractTensor getTensorForPosition(int layerIndex, int position, int index) {
            // Calculate page indices and relative indices
            int layerPageIndex = layerIndex / pageContext.layersPerPage;
            int contextPageIndex = position / pageContext.contextLengthPerPage;
            int relativeLayerIndex = layerIndex % pageContext.layersPerPage;
            int relativeContextIndex = position % pageContext.contextLengthPerPage;

            KvBufferPage page = pages[layerPageIndex][contextPageIndex];
            if (page == null || page.isClosed()) {
                page = new KvBufferPage(pageContext, "L" + layerPageIndex + "C" + contextPageIndex);
                pages[layerPageIndex][contextPageIndex] = page;
            }

            return page.getTensor().slice(true, relativeLayerIndex, index, relativeContextIndex);
        }

        public AbstractTensor[] getKeyTensorsUptoPosition(int layerIndex, int upperBound) {
            return getTensorsUptoPosition(layerIndex, 0, upperBound);
        }

        public AbstractTensor[] getValTensorsUptoPosition(int layerIndex, int upperBound) {
            return getTensorsUptoPosition(layerIndex, 1, upperBound);
        }

        private AbstractTensor[] getTensorsUptoPosition(int layerIndex, int index, int upperBound) {
            int layerPageIndex = layerIndex / pageContext.layersPerPage;
            int contextPageIndex = upperBound / pageContext.contextLengthPerPage;
            int relativeLayerIndex = layerIndex % pageContext.layersPerPage;

            KvBufferPage[] layerPages = pages[layerPageIndex];

            AbstractTensor[] tensors = new AbstractTensor[contextPageIndex + 1];

            for (int i = 0; i <= contextPageIndex; i++) {
                KvBufferPage page = layerPages[i];

                if (page == null || page.isClosed()) {
                    page = new KvBufferPage(pageContext, "L" + layerPageIndex + "C" + i);
                    layerPages[i] = page;
                }

                tensors[i] = page.getTensor().slice(true, relativeLayerIndex, index);
            }

            return tensors;
        }

        @Override
        public String toString() {
            return "KvBuffer{" +
                    "session=" + session +
                    ", currentContextPosition=" + currentContextPosition +
                    ", pages=" + Arrays.toString(pages) +
                    ", pageContext=" + pageContext +
                    '}';
        }
    }


}
