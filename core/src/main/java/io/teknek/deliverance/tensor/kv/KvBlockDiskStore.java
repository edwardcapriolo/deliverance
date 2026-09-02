package io.teknek.deliverance.tensor.kv;

import com.google.common.base.Preconditions;
import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.JsonUtils;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.impl.Q8ByteBufferTensor;

import javax.annotation.Nullable;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.lang.foreign.ValueLayout;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HexFormat;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.zip.CRC32;

/** Optional exact-layout disk persistence for immutable KVCache2 shared prefix blocks. */
public final class KvBlockDiskStore implements AutoCloseable {
    static final long MIN_DISK_CACHE_BYTES = 1024L * 1024L * 1024L;
    private static final String METRIC_PREFIX = "kvcache.v2.disk";
    private static final WriteTask POISON = new WriteTask(null, null, 0L);

    record Metadata(int version, KvBlockKey key, long payloadBytes, long checksumCrc32, long totalBytes,
            long lastAccessMillis) {
    }

    private record WriteTask(KvBlockKey key, byte[] payload, long checksumCrc32) {
    }

    private record DiskEntry(Path metaPath, Path binPath, Metadata metadata, long totalBytes) {
    }

    private final Path root;
    private final MetricRegistry metricRegistry;
    private final TensorAllocator allocator;
    private final long maxBytes;
    private final long reservedFreeBytes;
    private final int admitMinTokens;
    private final ArrayBlockingQueue<WriteTask> writeQueue;
    private final Thread writerThread;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    private KvBlockDiskStore(Path root, MetricRegistry metricRegistry, TensorAllocator allocator, long maxBytes,
            long reservedFreeBytes, int admitMinTokens, int queueSize) {
        this.root = root;
        this.metricRegistry = metricRegistry;
        this.allocator = allocator;
        this.maxBytes = maxBytes;
        this.reservedFreeBytes = reservedFreeBytes;
        this.admitMinTokens = admitMinTokens;
        this.writeQueue = new ArrayBlockingQueue<>(queueSize);
        this.writerThread = new Thread(this::writerLoop, "deliverance-kvcache-disk-writer");
        this.writerThread.setDaemon(true);
        this.writerThread.start();
    }

    @Nullable
    public static KvBlockDiskStore open(KvBufferCacheSettings settings, TensorAllocator allocator,
            MetricRegistry metricRegistry) {
        Objects.requireNonNull(settings, "settings");
        if (!settings.isSharedPrefixDiskCacheEnabled()) {
            return null;
        }
        if (settings.getPrefixCacheMode() != KvBufferCacheSettings.PrefixCacheMode.SHARED_BLOCKS) {
            InferenceProfiler.counter(metricRegistry, METRIC_PREFIX + ".disabled.mode").inc();
            return null;
        }
        if (settings.getSharedPrefixDiskCacheMaxBytes() < MIN_DISK_CACHE_BYTES) {
            InferenceProfiler.counter(metricRegistry, METRIC_PREFIX + ".disabled.max_bytes_too_small").inc();
            return null;
        }
        Path root = settings.getSharedPrefixDiskCachePath().toPath();
        try {
            Files.createDirectories(root);
        } catch (IOException e) {
            InferenceProfiler.counter(metricRegistry, METRIC_PREFIX + ".disabled.directory_error").inc();
            return null;
        }
        if (root.toFile().getUsableSpace() < settings.getSharedPrefixDiskCacheMinUsableBytes()) {
            InferenceProfiler.counter(metricRegistry, METRIC_PREFIX + ".disabled.usable_bytes_too_small").inc();
            return null;
        }
        KvBlockDiskStore store = new KvBlockDiskStore(root, metricRegistry, allocator,
                settings.getSharedPrefixDiskCacheMaxBytes(), settings.getSharedPrefixDiskCacheReservedFreeBytes(),
                settings.getSharedPrefixDiskCacheAdmitMinTokens(), settings.getSharedPrefixDiskCacheWriterQueueSize());
        store.evictToBudget();
        return store;
    }

    public KvBlock load(KvBlockKey key) {
        Objects.requireNonNull(key, "key");
        InferenceProfiler.counter(metricRegistry, METRIC_PREFIX + ".lookup").inc();
        long start = System.nanoTime();
        try {
            Path metaPath = metaPath(key);
            Path binPath = binPath(key);
            if (!Files.isRegularFile(metaPath) || !Files.isRegularFile(binPath)) {
                InferenceProfiler.counter(metricRegistry, METRIC_PREFIX + ".miss").inc();
                return null;
            }
            Metadata metadata = JsonUtils.om.readValue(metaPath.toFile(), Metadata.class);
            if (!key.equals(metadata.key()) || metadata.version() != 1) {
                InferenceProfiler.counter(metricRegistry, METRIC_PREFIX + ".miss.metadata_mismatch").inc();
                return null;
            }
            byte[] payload = Files.readAllBytes(binPath);
            InferenceProfiler.counter(metricRegistry, METRIC_PREFIX + ".bytes.read").inc(payload.length);
            if (payload.length != metadata.payloadBytes() || checksum(payload) != metadata.checksumCrc32()) {
                InferenceProfiler.counter(metricRegistry, METRIC_PREFIX + ".miss.checksum").inc();
                return null;
            }
            KvBlock block = deserializeDenseBlock(key, payload);
            touch(metaPath, binPath, metadata);
            InferenceProfiler.counter(metricRegistry, METRIC_PREFIX + ".hit").inc();
            return block;
        } catch (RuntimeException | IOException e) {
            InferenceProfiler.counter(metricRegistry, METRIC_PREFIX + ".miss.error").inc();
            return null;
        } finally {
            InferenceProfiler.timer(metricRegistry, METRIC_PREFIX + ".load.elapsed")
                    .update(System.nanoTime() - start, TimeUnit.NANOSECONDS);
        }
    }

    public void enqueueWrite(KvBlockKey key, KvBlock block) {
        enqueueWrite(key, block, key.tokenCount());
    }

    public void enqueueWrite(KvBlockKey key, KvBlock block, int admitTokenCount) {
        if (closed.get()) {
            return;
        }
        if (admitTokenCount < admitMinTokens) {
            InferenceProfiler.counter(metricRegistry, METRIC_PREFIX + ".write.skipped.too_small").inc();
            return;
        }
        if (block.layout() != KvBlockLayout.DENSE || !(block.storage() instanceof DenseKvBlockStorage)) {
            InferenceProfiler.counter(metricRegistry, METRIC_PREFIX + ".write.skipped.unsupported_layout").inc();
            return;
        }
        if (Files.exists(metaPath(key)) && Files.exists(binPath(key))) {
            InferenceProfiler.counter(metricRegistry, METRIC_PREFIX + ".write.skipped.exists").inc();
            return;
        }
        byte[] payload = serializeDenseBlock(block);
        if (root.toFile().getUsableSpace() - payload.length < reservedFreeBytes) {
            InferenceProfiler.counter(metricRegistry, METRIC_PREFIX + ".write.skipped.low_space").inc();
            return;
        }
        if (!writeQueue.offer(new WriteTask(key, payload, checksum(payload)))) {
            InferenceProfiler.counter(metricRegistry, METRIC_PREFIX + ".write.skipped.queue_full").inc();
        }
    }

    private void writerLoop() {
        while (true) {
            try {
                WriteTask task = writeQueue.take();
                if (task == POISON) {
                    return;
                }
                write(task);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return;
            }
        }
    }

    private void write(WriteTask task) {
        long start = System.nanoTime();
        Path metaPath = metaPath(task.key());
        Path binPath = binPath(task.key());
        if (Files.exists(metaPath) && Files.exists(binPath)) {
            InferenceProfiler.counter(metricRegistry, METRIC_PREFIX + ".write.skipped.exists").inc();
            return;
        }
        Path tmpMeta = metaPath.resolveSibling(metaPath.getFileName() + ".tmp");
        Path tmpBin = binPath.resolveSibling(binPath.getFileName() + ".tmp");
        try {
            Files.createDirectories(root);
            Metadata metadata = new Metadata(1, task.key(), task.payload().length, task.checksumCrc32(), 0L,
                    System.currentTimeMillis());
            byte[] metaBytes = JsonUtils.om.writeValueAsBytes(metadata);
            metadata = new Metadata(1, task.key(), task.payload().length, task.checksumCrc32(),
                    task.payload().length + metaBytes.length, System.currentTimeMillis());
            metaBytes = JsonUtils.om.writeValueAsBytes(metadata);
            writeFile(tmpBin, task.payload());
            writeFile(tmpMeta, metaBytes);
            Files.move(tmpBin, binPath, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
            Files.move(tmpMeta, metaPath, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
            InferenceProfiler.counter(metricRegistry, METRIC_PREFIX + ".bytes.written").inc(task.payload().length);
            InferenceProfiler.counter(metricRegistry, METRIC_PREFIX + ".write").inc();
            evictToBudget();
        } catch (IOException e) {
            InferenceProfiler.counter(metricRegistry, METRIC_PREFIX + ".write.error").inc();
            deleteQuietly(tmpMeta);
            deleteQuietly(tmpBin);
        } finally {
            InferenceProfiler.timer(metricRegistry, METRIC_PREFIX + ".write.elapsed")
                    .update(System.nanoTime() - start, TimeUnit.NANOSECONDS);
        }
    }

    private byte[] serializeDenseBlock(KvBlock block) {
        DenseKvBlockStorage storage = (DenseKvBlockStorage) block.storage();
        try {
            ByteArrayOutputStream out = new ByteArrayOutputStream(Math.toIntExact(block.encodedBytes()));
            writeTensor(out, storage.keyStorage());
            writeTensor(out, storage.valueStorage());
            return out.toByteArray();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private KvBlock deserializeDenseBlock(KvBlockKey key, byte[] payload) throws IOException {
        int cursor = 0;
        AbstractTensor keyStorage = allocator.getDirty(key.keyDType(), TensorShape.of(key.layers(), key.blockSize(), key.kvLength()));
        AbstractTensor valueStorage = allocator.getDirty(key.valueDType(), TensorShape.of(key.layers(), key.blockSize(), key.kvLength()));
        try {
            cursor = readTensor(payload, cursor, keyStorage);
            readTensor(payload, cursor, valueStorage);
            return new KvBlock(key.blockIndex(), key.blockSize(), key.tokenCount(), key.layers(), key.kvLength(),
                    new DenseKvBlockStorage(key.layers(), key.tokenCount(), key.blockSize(), key.kvLength(), keyStorage,
                            valueStorage));
        } catch (RuntimeException | IOException e) {
            keyStorage.close();
            valueStorage.close();
            throw e;
        }
    }

    private void writeTensor(ByteArrayOutputStream out, AbstractTensor tensor) throws IOException {
        int bytes = Math.toIntExact(tensor.size() * tensor.dType().size());
        out.write(tensor.getMemorySegment().asSlice(tensor.getMemorySegmentOffset(0), bytes).toArray(ValueLayout.JAVA_BYTE));
        if (tensor instanceof Q8ByteBufferTensor q8) {
            AbstractTensor scale = q8.getBlockF();
            int scaleBytes = Math.toIntExact(scale.size() * scale.dType().size());
            out.write(scale.getMemorySegment().asSlice(scale.getMemorySegmentOffset(0), scaleBytes)
                    .toArray(ValueLayout.JAVA_BYTE));
        }
    }

    private int readTensor(byte[] payload, int cursor, AbstractTensor tensor) throws IOException {
        int bytes = Math.toIntExact(tensor.size() * tensor.dType().size());
        copyIntoTensor(payload, cursor, tensor, bytes);
        cursor += bytes;
        if (tensor instanceof Q8ByteBufferTensor q8) {
            AbstractTensor scale = q8.getBlockF();
            int scaleBytes = Math.toIntExact(scale.size() * scale.dType().size());
            copyIntoTensor(payload, cursor, scale, scaleBytes);
            cursor += scaleBytes;
        }
        return cursor;
    }

    private void copyIntoTensor(byte[] payload, int cursor, AbstractTensor tensor, int bytes) throws IOException {
        if (cursor + bytes > payload.length) {
            throw new IOException("KV disk payload is truncated");
        }
        tensor.getMemorySegment().asSlice(tensor.getMemorySegmentOffset(0), bytes)
                .copyFrom(java.lang.foreign.MemorySegment.ofArray(payload).asSlice(cursor, bytes));
    }

    private void touch(Path metaPath, Path binPath, Metadata metadata) throws IOException {
        Metadata touched = new Metadata(metadata.version(), metadata.key(), metadata.payloadBytes(), metadata.checksumCrc32(),
                Files.size(metaPath) + Files.size(binPath), System.currentTimeMillis());
        writeFile(metaPath, JsonUtils.om.writeValueAsBytes(touched));
    }

    private void evictToBudget() {
        try {
            List<DiskEntry> entries = entries();
            long total = entries.stream().mapToLong(DiskEntry::totalBytes).sum();
            entries.sort(Comparator.comparingLong(entry -> entry.metadata().lastAccessMillis()));
            for (DiskEntry entry : entries) {
                if (total <= maxBytes) {
                    return;
                }
                deleteQuietly(entry.metaPath());
                deleteQuietly(entry.binPath());
                total -= entry.totalBytes();
                InferenceProfiler.counter(metricRegistry, METRIC_PREFIX + ".evict").inc();
                InferenceProfiler.counter(metricRegistry, METRIC_PREFIX + ".evict.bytes").inc(entry.totalBytes());
            }
        } catch (IOException e) {
            InferenceProfiler.counter(metricRegistry, METRIC_PREFIX + ".evict.error").inc();
        }
    }

    private List<DiskEntry> entries() throws IOException {
        if (!Files.isDirectory(root)) {
            return List.of();
        }
        ArrayList<DiskEntry> entries = new ArrayList<>();
        try (var stream = Files.list(root)) {
            for (Path metaPath : stream.filter(path -> path.getFileName().toString().endsWith(".meta.json")).toList()) {
                try {
                    Metadata metadata = JsonUtils.om.readValue(metaPath.toFile(), Metadata.class);
                    Path binPath = binPath(metadata.key());
                    if (!Files.isRegularFile(binPath)) {
                        continue;
                    }
                    entries.add(new DiskEntry(metaPath, binPath, metadata, Files.size(metaPath) + Files.size(binPath)));
                } catch (RuntimeException | IOException e) {
                    InferenceProfiler.counter(metricRegistry, METRIC_PREFIX + ".scan.skip_corrupt").inc();
                }
            }
        }
        return entries;
    }

    private Path metaPath(KvBlockKey key) {
        return root.resolve(keyHash(key) + ".meta.json");
    }

    private Path binPath(KvBlockKey key) {
        return root.resolve(keyHash(key) + ".bin");
    }

    private String keyHash(KvBlockKey key) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            return HexFormat.of().formatHex(digest.digest(JsonUtils.om.writeValueAsBytes(key)));
        } catch (NoSuchAlgorithmException | IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static long checksum(byte[] payload) {
        CRC32 crc = new CRC32();
        crc.update(payload);
        return crc.getValue();
    }

    private void writeFile(Path path, byte[] bytes) throws IOException {
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING,
                StandardOpenOption.WRITE)) {
            channel.write(java.nio.ByteBuffer.wrap(bytes));
            channel.force(true);
        }
    }

    private static void deleteQuietly(Path path) {
        try {
            Files.deleteIfExists(path);
        } catch (IOException ignored) {
        }
    }

    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            while (!writeQueue.offer(POISON)) {
                writeQueue.poll();
            }
            try {
                writerThread.join(1000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }
}
