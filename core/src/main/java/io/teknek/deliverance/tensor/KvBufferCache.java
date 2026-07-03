package io.teknek.deliverance.tensor;

import com.google.common.base.Preconditions;
import io.teknek.deliverance.DType;

import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.tensor.impl.BFloat16BufferTensor;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import net.jpountz.lz4.LZ4Compressor;
import net.jpountz.lz4.LZ4Exception;
import net.jpountz.lz4.LZ4Factory;
import net.jpountz.lz4.LZ4FastDecompressor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.lang.foreign.MemorySegment;
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
    public record PrefixEntry(KvBuffer buffer, int length, boolean temporary) {
        public PrefixEntry(KvBuffer buffer, int length) {
            this(buffer, length, false);
        }

        public void closeIfTemporary() {
            if (temporary && buffer != null) {
                buffer.close();
            }
        }
    }

    public interface StoredPrefixEntry extends AutoCloseable {
        PrefixEntry toPrefixEntry();

        int length();

        @Override
        void close();
    }

    private record RawStoredPrefixEntry(KvBuffer buffer, int length) implements StoredPrefixEntry {
        @Override
        public PrefixEntry toPrefixEntry() {
            return new PrefixEntry(buffer, length, false);
        }

        @Override
        public void close() {
            buffer.close();
        }
    }

    private record Lz4StoredPrefixEntry(byte[] compressed, int uncompressedBytes, int length) implements StoredPrefixEntry {
        @Override
        public PrefixEntry toPrefixEntry() {
            throw new UnsupportedOperationException("LZ4 prefix entries require cache context to hydrate");
        }

        @Override
        public void close() {
            // compressed byte arrays are owned by the JVM GC
        }
    }

    private record MseTurboQuantStoredPrefixEntry(byte[] packedCodes, float[] norms, int length, int bitWidth,
            int kvLength, int rotatedDim) implements StoredPrefixEntry {
        @Override
        public PrefixEntry toPrefixEntry() {
            throw new UnsupportedOperationException("MSE TurboQuant prefix entries require cache context to hydrate");
        }

        @Override
        public void close() {
            // packed code and norm arrays are owned by the JVM GC
        }
    }

    private static final Logger logger = LoggerFactory.getLogger(KvBufferCache.class);
    private static final LZ4Factory LZ4_FACTORY = LZ4Factory.fastestInstance();
    private static final long TURBO_ROTATION_SEED = 0x6A09E667F3BCC909L;
    private static final Map<Integer, float[]> TURBO_CODEBOOKS = Collections.synchronizedMap(new HashMap<>());

    private final AbstractModel model;
    private final KvBufferCacheSettings kvBufferCacheSettings;
    private final int blockSize;
    private final Set<Path> activeDiskPages = Collections.synchronizedSet(new HashSet<>());
    private final Map<Path, Long> diskPageBytes = Collections.synchronizedMap(new HashMap<>());
    private final AtomicBoolean closed = new AtomicBoolean(false);
    private ScheduledExecutorService diskSweeperExecutor;

    public final Map<CacheKey, StoredPrefixEntry> prefixCache = Collections.synchronizedMap(
            new LinkedHashMap<CacheKey, StoredPrefixEntry>(16, 0.75f, true) {
                public boolean removeEldestEntry(Map.Entry<CacheKey, StoredPrefixEntry> eldest) {
                    boolean evict = size() > kvBufferCacheSettings.getMaxEntries();
                    if (evict && eldest != null && eldest.getValue() != null) {
                        model.getMetricRegistry().meter("kvbuffercache.evict").mark();
                        try {
                            eldest.getValue().close();
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
        StoredPrefixEntry best = null;
        int limit = kvBufferCacheSettings.getMaxPrefixTokensPerPrompt();
        for (int prefixLen : checkpointLengths(Math.min(tokens.length, limit))) {
            StoredPrefixEntry e = prefixCache.get(new CacheKey(salt, prefixTokens(tokens, prefixLen)));
            if (e != null) {
                best = e;
            }
        }
        model.getMetricRegistry().meter("kvbuffercache.lookup").mark();
        if (best != null && best.length() >= blockSize && best.length() % blockSize == 0) {
            model.getMetricRegistry().meter("kvbuffercache.hits").mark();
            return toPrefixEntry(best);
        }
        model.getMetricRegistry().meter("kvbuffercache.misses").mark();
        return null;
    }

    public void storePrefix(int[] tokens, KvBuffer buffer, Optional<String> salt) {
        long storeStart = System.nanoTime();
        if (!kvBufferCacheSettings.isEphemeral()) {
            model.getMetricRegistry().meter("kvbuffercache.prefix.disk.skip").mark();
            return;
        }
        if (kvBufferCacheSettings.getMaxEntries() < 1){
            return;
        }
        try {
        int limit = kvBufferCacheSettings.getMaxPrefixTokensPerPrompt();
        for (int prefixLen : checkpointLengths(Math.min(tokens.length, limit))) {
            KvBuffer snapshot = getEphemeralKvBuffer();
            copyPrefix(buffer, snapshot, prefixLen);
            StoredPrefixEntry entry = toStoredPrefixEntry(snapshot, prefixLen);
            StoredPrefixEntry previous = prefixCache.putIfAbsent(new CacheKey(salt, prefixTokens(tokens, prefixLen)), entry);
            if (previous != null) {
                entry.close();
            }
        }
        } finally {
            InferenceProfiler.timer(model.getMetricRegistry(), "kvbuffercache.prefix.store")
                    .update(System.nanoTime() - storeStart, TimeUnit.NANOSECONDS);
        }
    }

    private PrefixEntry toPrefixEntry(StoredPrefixEntry stored) {
        if (stored instanceof RawStoredPrefixEntry raw) {
            return raw.toPrefixEntry();
        }
        if (stored instanceof Lz4StoredPrefixEntry lz4) {
            return hydrateLz4PrefixEntry(lz4);
        }
        if (stored instanceof MseTurboQuantStoredPrefixEntry turboQuant) {
            return hydrateMseTurboQuantPrefixEntry(turboQuant);
        }
        throw new IllegalStateException("unknown prefix entry type " + stored.getClass());
    }

    private StoredPrefixEntry toStoredPrefixEntry(KvBuffer snapshot, int prefixLen) {
        if (kvBufferCacheSettings.getPrefixCompression() == KvBufferCacheSettings.PrefixCompression.NONE) {
            return new RawStoredPrefixEntry(snapshot, prefixLen);
        }
        if (kvBufferCacheSettings.getPrefixCompression() == KvBufferCacheSettings.PrefixCompression.LZ4) {
            try {
                return compressPrefixEntry(snapshot, prefixLen);
            } finally {
                snapshot.close();
            }
        }
        if (kvBufferCacheSettings.getPrefixCompression() == KvBufferCacheSettings.PrefixCompression.MSE_TURBOQUANT) {
            try {
                return encodeMseTurboQuantPrefixEntry(snapshot, prefixLen);
            } finally {
                snapshot.close();
            }
        }
        throw new UnsupportedOperationException("Unsupported prefix compression " + kvBufferCacheSettings.getPrefixCompression());
    }

    private Lz4StoredPrefixEntry compressPrefixEntry(KvBuffer snapshot, int prefixLen) {
        long start = System.nanoTime();
        byte[] raw = serializePrefix(snapshot, prefixLen);
        LZ4Compressor compressor = LZ4_FACTORY.fastCompressor();
        byte[] compressed = compressor.compress(raw);
        InferenceProfiler.timer(model.getMetricRegistry(), "kvbuffercache.prefix.lz4.compress")
                .update(System.nanoTime() - start, TimeUnit.NANOSECONDS);
        InferenceProfiler.counter(model.getMetricRegistry(), "kvbuffercache.prefix.lz4.uncompressed.bytes").inc(raw.length);
        InferenceProfiler.counter(model.getMetricRegistry(), "kvbuffercache.prefix.lz4.compressed.bytes").inc(compressed.length);
        return new Lz4StoredPrefixEntry(compressed, raw.length, prefixLen);
    }

    private PrefixEntry hydrateLz4PrefixEntry(Lz4StoredPrefixEntry stored) {
        long start = System.nanoTime();
        byte[] raw = new byte[stored.uncompressedBytes()];
        try {
            LZ4FastDecompressor decompressor = LZ4_FACTORY.fastDecompressor();
            decompressor.decompress(stored.compressed(), 0, raw, 0, raw.length);
        } catch (LZ4Exception e) {
            model.getMetricRegistry().meter("kvbuffercache.prefix.lz4.decompress.error").mark();
            throw new IllegalStateException("Unable to decompress prefix cache entry", e);
        }
        KvBuffer hydrated = getEphemeralKvBuffer();
        deserializePrefix(raw, hydrated, stored.length());
        InferenceProfiler.timer(model.getMetricRegistry(), "kvbuffercache.prefix.lz4.decompress")
                .update(System.nanoTime() - start, TimeUnit.NANOSECONDS);
        return new PrefixEntry(hydrated, stored.length(), true);
    }

    private MseTurboQuantStoredPrefixEntry encodeMseTurboQuantPrefixEntry(KvBuffer snapshot, int prefixLen) {
        long start = System.nanoTime();
        int bitWidth = kvBufferCacheSettings.getPrefixTurboQuantBits();
        int kvLength = model.getLocalKvLength();
        int rotatedDim = nextPowerOfTwo(kvLength);
        int layers = model.getConfig().numberOfLayers;
        int rows = Math.multiplyExact(Math.multiplyExact(layers, prefixLen), 2);
        long totalCodes = Math.multiplyExact((long) rows, rotatedDim);
        byte[] packedCodes = new byte[Math.toIntExact((totalCodes * bitWidth + 7) / 8)];
        float[] norms = new float[rows];
        float[] codebook = turboCodebook(bitWidth);
        float invSqrtDim = (float) (1.0 / Math.sqrt(rotatedDim));
        int rowIndex = 0;
        for (int layer = 0; layer < layers; layer++) {
            for (int pos = 0; pos < prefixLen; pos++) {
                rowIndex = encodeMseTurboQuantRow(snapshot.getKeyTensorForPosition(layer, pos), packedCodes, norms,
                        rowIndex, bitWidth, kvLength, rotatedDim, invSqrtDim, codebook);
                rowIndex = encodeMseTurboQuantRow(snapshot.getValTensorForPosition(layer, pos), packedCodes, norms,
                        rowIndex, bitWidth, kvLength, rotatedDim, invSqrtDim, codebook);
            }
        }
        long rawBytes = (long) rows * kvLength * model.getWorkingDType().size();
        long encodedBytes = packedCodes.length + (long) norms.length * Float.BYTES;
        InferenceProfiler.timer(model.getMetricRegistry(), "kvbuffercache.prefix.turboquant.encode")
                .update(System.nanoTime() - start, TimeUnit.NANOSECONDS);
        InferenceProfiler.counter(model.getMetricRegistry(), "kvbuffercache.prefix.turboquant.raw.bytes").inc(rawBytes);
        InferenceProfiler.counter(model.getMetricRegistry(), "kvbuffercache.prefix.turboquant.encoded.bytes").inc(encodedBytes);
        return new MseTurboQuantStoredPrefixEntry(packedCodes, norms, prefixLen, bitWidth, kvLength, rotatedDim);
    }

    private int encodeMseTurboQuantRow(AbstractTensor row, byte[] packedCodes, float[] norms, int rowIndex,
            int bitWidth, int kvLength, int rotatedDim, float invSqrtDim, float[] codebook) {
        try (row) {
            float normSquared = 0.0f;
            for (int i = 0; i < kvLength; i++) {
                float value = row.get(0, i);
                normSquared += value * value;
            }
            float norm = (float) Math.sqrt(normSquared);
            norms[rowIndex] = norm;
            float[] rotated = new float[rotatedDim];
            if (norm != 0.0f) {
                float inverseNorm = 1.0f / norm;
                for (int i = 0; i < kvLength; i++) {
                    rotated[i] = row.get(0, i) * inverseNorm * turboSign(i);
                }
                fastWalshHadamard(rotated);
                for (int i = 0; i < rotatedDim; i++) {
                    rotated[i] *= invSqrtDim;
                }
            }
            long baseCode = (long) rowIndex * rotatedDim;
            float coordinateScale = (float) Math.sqrt(rotatedDim);
            for (int i = 0; i < rotatedDim; i++) {
                int code = nearestCodebookIndex(rotated[i] * coordinateScale, codebook);
                packCode(packedCodes, baseCode + i, bitWidth, code);
            }
            return rowIndex + 1;
        }
    }

    private PrefixEntry hydrateMseTurboQuantPrefixEntry(MseTurboQuantStoredPrefixEntry stored) {
        long start = System.nanoTime();
        KvBuffer hydrated = getEphemeralKvBuffer();
        float[] codebook = turboCodebook(stored.bitWidth());
        float invSqrtDim = (float) (1.0 / Math.sqrt(stored.rotatedDim()));
        int layers = model.getConfig().numberOfLayers;
        int rowIndex = 0;
        for (int layer = 0; layer < layers; layer++) {
            for (int pos = 0; pos < stored.length(); pos++) {
                rowIndex = decodeMseTurboQuantRow(stored, hydrated.getKeyTensorForPosition(layer, pos), rowIndex,
                        invSqrtDim, codebook);
                rowIndex = decodeMseTurboQuantRow(stored, hydrated.getValTensorForPosition(layer, pos), rowIndex,
                        invSqrtDim, codebook);
            }
        }
        hydrated.setCurrentContextPosition(stored.length());
        InferenceProfiler.timer(model.getMetricRegistry(), "kvbuffercache.prefix.turboquant.decode")
                .update(System.nanoTime() - start, TimeUnit.NANOSECONDS);
        return new PrefixEntry(hydrated, stored.length(), true);
    }

    private int decodeMseTurboQuantRow(MseTurboQuantStoredPrefixEntry stored, AbstractTensor row, int rowIndex,
            float invSqrtDim, float[] codebook) {
        try (row) {
            float[] rotated = new float[stored.rotatedDim()];
            long baseCode = (long) rowIndex * stored.rotatedDim();
            for (int i = 0; i < stored.rotatedDim(); i++) {
                int code = unpackCode(stored.packedCodes(), baseCode + i, stored.bitWidth());
                rotated[i] = codebook[code] * invSqrtDim;
            }
            fastWalshHadamard(rotated);
            float normScale = stored.norms()[rowIndex] * invSqrtDim;
            for (int i = 0; i < stored.kvLength(); i++) {
                row.set(rotated[i] * normScale * turboSign(i), 0, i);
            }
            return rowIndex + 1;
        }
    }

    private static int nextPowerOfTwo(int value) {
        if (value < 1) {
            throw new IllegalArgumentException("value must be positive");
        }
        int highest = Integer.highestOneBit(value);
        return highest == value ? value : highest << 1;
    }

    private static float turboSign(int index) {
        long x = TURBO_ROTATION_SEED + 0x9E3779B97F4A7C15L * index;
        x = (x ^ (x >>> 30)) * 0xBF58476D1CE4E5B9L;
        x = (x ^ (x >>> 27)) * 0x94D049BB133111EBL;
        x = x ^ (x >>> 31);
        return (x & 1L) == 0L ? 1.0f : -1.0f;
    }

    private static void fastWalshHadamard(float[] values) {
        for (int step = 1; step < values.length; step <<= 1) {
            for (int base = 0; base < values.length; base += step << 1) {
                for (int i = 0; i < step; i++) {
                    float a = values[base + i];
                    float b = values[base + i + step];
                    values[base + i] = a + b;
                    values[base + i + step] = a - b;
                }
            }
        }
    }

    private static float[] turboCodebook(int bitWidth) {
        return TURBO_CODEBOOKS.computeIfAbsent(bitWidth, KvBufferCache::buildNormalLloydMaxCodebook);
    }

    private static float[] buildNormalLloydMaxCodebook(int bitWidth) {
        int levels = 1 << bitWidth;
        float[] centroids = new float[levels];
        float min = -6.0f;
        float max = 6.0f;
        for (int i = 0; i < levels; i++) {
            centroids[i] = levels == 1 ? 0.0f : min + (max - min) * (i + 0.5f) / levels;
        }
        int samples = 20_001;
        float[] sampleValues = new float[samples];
        float[] sampleWeights = new float[samples];
        float dx = (max - min) / (samples - 1);
        for (int i = 0; i < samples; i++) {
            float x = min + i * dx;
            sampleValues[i] = x;
            sampleWeights[i] = (float) Math.exp(-0.5f * x * x);
        }
        for (int iter = 0; iter < 80; iter++) {
            float[] weightedSums = new float[levels];
            float[] weights = new float[levels];
            for (int i = 0; i < samples; i++) {
                int nearest = nearestCodebookIndex(sampleValues[i], centroids);
                weightedSums[nearest] += sampleValues[i] * sampleWeights[i];
                weights[nearest] += sampleWeights[i];
            }
            for (int i = 0; i < levels; i++) {
                if (weights[i] > 0.0f) {
                    centroids[i] = weightedSums[i] / weights[i];
                }
            }
            Arrays.sort(centroids);
        }
        return centroids;
    }

    private static int nearestCodebookIndex(float value, float[] codebook) {
        int best = 0;
        float bestDistance = Math.abs(value - codebook[0]);
        for (int i = 1; i < codebook.length; i++) {
            float distance = Math.abs(value - codebook[i]);
            if (distance < bestDistance) {
                bestDistance = distance;
                best = i;
            }
        }
        return best;
    }

    private static void packCode(byte[] packed, long codeIndex, int bitWidth, int code) {
        long bitOffset = codeIndex * bitWidth;
        for (int bit = 0; bit < bitWidth; bit++) {
            if (((code >>> bit) & 1) != 0) {
                long absoluteBit = bitOffset + bit;
                int byteIndex = Math.toIntExact(absoluteBit >>> 3);
                packed[byteIndex] = (byte) (packed[byteIndex] | (1 << (absoluteBit & 7)));
            }
        }
    }

    private static int unpackCode(byte[] packed, long codeIndex, int bitWidth) {
        long bitOffset = codeIndex * bitWidth;
        int code = 0;
        for (int bit = 0; bit < bitWidth; bit++) {
            long absoluteBit = bitOffset + bit;
            int byteIndex = Math.toIntExact(absoluteBit >>> 3);
            if (((packed[byteIndex] >>> (absoluteBit & 7)) & 1) != 0) {
                code |= 1 << bit;
            }
        }
        return code;
    }

    private byte[] serializePrefix(KvBuffer buffer, int prefixLen) {
        int rowBytes = model.getLocalKvLength() * model.getWorkingDType().size();
        int layers = model.getConfig().numberOfLayers;
        byte[] bytes = new byte[Math.multiplyExact(Math.multiplyExact(layers, prefixLen), rowBytes * 2)];
        MemorySegment destination = MemorySegment.ofArray(bytes);
        int offset = 0;
        for (int layer = 0; layer < layers; layer++) {
            for (int pos = 0; pos < prefixLen; pos++) {
                offset = copyRowBytes(buffer.getKeyTensorForPosition(layer, pos), destination, offset, rowBytes);
                offset = copyRowBytes(buffer.getValTensorForPosition(layer, pos), destination, offset, rowBytes);
            }
        }
        return bytes;
    }

    private int copyRowBytes(AbstractTensor row, MemorySegment destination, int offset, int rowBytes) {
        try (row) {
            destination.asSlice(offset, rowBytes)
                    .copyFrom(row.getMemorySegment().asSlice(row.getMemorySegmentOffset(0), rowBytes));
            return offset + rowBytes;
        }
    }

    private void deserializePrefix(byte[] bytes, KvBuffer buffer, int prefixLen) {
        int rowBytes = model.getLocalKvLength() * model.getWorkingDType().size();
        int layers = model.getConfig().numberOfLayers;
        MemorySegment source = MemorySegment.ofArray(bytes);
        int offset = 0;
        for (int layer = 0; layer < layers; layer++) {
            for (int pos = 0; pos < prefixLen; pos++) {
                offset = restoreRowBytes(source, offset, buffer.getKeyTensorForPosition(layer, pos), rowBytes);
                offset = restoreRowBytes(source, offset, buffer.getValTensorForPosition(layer, pos), rowBytes);
            }
        }
        buffer.setCurrentContextPosition(prefixLen);
    }

    private int restoreRowBytes(MemorySegment source, int offset, AbstractTensor row, int rowBytes) {
        try (row) {
            row.getMemorySegment().asSlice(row.getMemorySegmentOffset(0), rowBytes)
                    .copyFrom(source.asSlice(offset, rowBytes));
            return offset + rowBytes;
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
        long copyStart = System.nanoTime();
        long copiedBytes = 0;
        Config c = model.getConfig();
        int layers = c.numberOfLayers;
        for (int layer = 0; layer < layers; layer++) {
            for (int pos = 0; pos < length; pos++) {
                AbstractTensor srcK = src.getKeyTensorForPosition(layer, pos);
                AbstractTensor srcV = src.getValTensorForPosition(layer, pos);

                AbstractTensor dstK = dest.getKeyTensorForPosition(layer, pos);
                AbstractTensor dstV = dest.getValTensorForPosition(layer, pos);
                copiedBytes += (long) srcK.size() * model.getWorkingDType().size();
                copiedBytes += (long) srcV.size() * model.getWorkingDType().size();
                dstK.copyFrom(srcK, 0, 0, (int) srcK.size());
                dstV.copyFrom(srcV, 0, 0, (int) srcV.size());
                srcK.close();
                srcV.close();
                dstK.close();
                dstV.close();
            }
        }
        dest.setCurrentContextPosition(length);
        InferenceProfiler.timer(model.getMetricRegistry(), "kvbuffercache.prefix.copy")
                .update(System.nanoTime() - copyStart, TimeUnit.NANOSECONDS);
        InferenceProfiler.counter(model.getMetricRegistry(), "kvbuffercache.prefix.copy.bytes").inc(copiedBytes);
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
            prefixCache.entrySet().iterator().forEachRemaining(e -> e.getValue().close());
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

                    AbstractTensor t;
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
