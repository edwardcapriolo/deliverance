package io.teknek.deliverance.tensor;

import javax.annotation.Nullable;
import java.io.File;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;

public class KvBufferCacheSettings {
    private final Boolean useTensorAllocator;
    private final File workingDirectory;
    private final TensorAllocator dedicatedCache;
    private boolean deleteDiskPagesOnClose = true;
    private boolean diskPageSweeperEnabled = true;
    private Duration diskPageSweepInterval = Duration.ofMinutes(5);
    private Duration diskPageMaxAge = Duration.ofHours(1);
    /**
     * The longest possible kvcache
     */
    private int maxPrefixTokensPerPrompt = 512;
    private PrefixCheckpointPolicy prefixCheckpointPolicy = PrefixCheckpointPolicy.ANCHORS_AND_LARGEST;
    private int maxPrefixCheckpointsPerPrompt = 4;
    private List<Integer> prefixCheckpointAnchors = List.of(32, 64, 128);
    /** the maximum size of the cache before evictions happen **/
    private int maxEntries = 10_000;
    /**
    The block size of the kvcache. Cache hits will only happen at block boundaries, smaller blockize uses more memory
     */
    private int blockSize = 32;
    /**
     * Preferred active KV page length in context-token rows.
     *
     * <p>During attention, Deliverance reads one layer's KV history across the context axis. If pages hold only a few
     * context rows, each token has to loop over many small key/value page slices. That increases Java loop overhead and
     * makes the score/value kernels work on tiny fragments. This target asks the KV cache to prefer pages with this many
     * adjacent context rows, then fit as many layers as possible into the configured page byte budget.</p>
     *
     * <p>The value is a target, not a guarantee. Large models with wide KV rows may fit fewer rows. Smaller models may fit
     * more layers per page while still using this row target. The default of {@code 32} was chosen because it materially
     * reduced Qwen3-4B page fan-out; larger 64-row and 128-row pages reduced page count further but did not improve the
     * representative benchmark.</p>
     */
    private int contextRowsPerPageTarget = 32;

    public enum KvFormat {
        BF16,
        F32,
        QUANTIZED_INT8
    }

    public enum PrefixCheckpointPolicy {
        FIXED_BLOCKS,
        ANCHORS_AND_LARGEST
    }

    private KvFormat kvFormat = KvFormat.BF16;
    /**
     * Use the tensor cache shared with model
     */
    public KvBufferCacheSettings(boolean ephemeral) {
        this.useTensorAllocator = ephemeral;
        this.dedicatedCache = null;
        this.workingDirectory = null;
    }

    /**
     * Use a dedicated tensor cache
     */
    public KvBufferCacheSettings(TensorAllocator cache){
        this.useTensorAllocator = true;
        this.dedicatedCache = cache;
        this.workingDirectory = null;
    }

    public KvBufferCacheSettings(File workingDirectory) {
        if (workingDirectory == null) {
            throw new IllegalArgumentException("workingDirectory must not be null");
        }
        this.workingDirectory = workingDirectory;
        this.useTensorAllocator = false;
        this.dedicatedCache = null;
    }

    public boolean isEphemeral() {
        return Boolean.TRUE.equals(useTensorAllocator);
    }

    @Nullable
    public File getWorkingDirectory() {
        return this.workingDirectory;
    }

    @Nullable
    public TensorAllocator getDedicatedCache(){
        return this.dedicatedCache;
    }

    public int getMaxPrefixTokensPerPrompt() {
        return maxPrefixTokensPerPrompt;
    }

    public void setMaxPrefixTokensPerPrompt(int maxPrefixTokensPerPrompt) {
        this.maxPrefixTokensPerPrompt = maxPrefixTokensPerPrompt;
    }
    public KvBufferCacheSettings withMaxPrefixTokensPerPrompt(int maxEntries) {
        this.maxPrefixTokensPerPrompt = maxEntries;
        return this;
    }

    public PrefixCheckpointPolicy getPrefixCheckpointPolicy() {
        return prefixCheckpointPolicy;
    }

    public void setPrefixCheckpointPolicy(PrefixCheckpointPolicy prefixCheckpointPolicy) {
        if (prefixCheckpointPolicy == null) {
            throw new IllegalArgumentException("prefixCheckpointPolicy must not be null");
        }
        this.prefixCheckpointPolicy = prefixCheckpointPolicy;
    }

    public KvBufferCacheSettings withPrefixCheckpointPolicy(PrefixCheckpointPolicy prefixCheckpointPolicy) {
        setPrefixCheckpointPolicy(prefixCheckpointPolicy);
        return this;
    }

    public int getMaxPrefixCheckpointsPerPrompt() {
        return maxPrefixCheckpointsPerPrompt;
    }

    public void setMaxPrefixCheckpointsPerPrompt(int maxPrefixCheckpointsPerPrompt) {
        if (maxPrefixCheckpointsPerPrompt < 1) {
            throw new IllegalArgumentException("maxPrefixCheckpointsPerPrompt must be >= 1");
        }
        this.maxPrefixCheckpointsPerPrompt = maxPrefixCheckpointsPerPrompt;
    }

    public KvBufferCacheSettings withMaxPrefixCheckpointsPerPrompt(int maxPrefixCheckpointsPerPrompt) {
        setMaxPrefixCheckpointsPerPrompt(maxPrefixCheckpointsPerPrompt);
        return this;
    }

    public List<Integer> getPrefixCheckpointAnchors() {
        return prefixCheckpointAnchors;
    }

    public void setPrefixCheckpointAnchors(List<Integer> prefixCheckpointAnchors) {
        if (prefixCheckpointAnchors == null || prefixCheckpointAnchors.isEmpty()) {
            throw new IllegalArgumentException("prefixCheckpointAnchors must not be empty");
        }
        ArrayList<Integer> copy = new ArrayList<>();
        for (Integer anchor : prefixCheckpointAnchors) {
            if (anchor == null || anchor < 1) {
                throw new IllegalArgumentException("prefixCheckpointAnchors must contain positive integers");
            }
            copy.add(anchor);
        }
        copy.sort(Integer::compareTo);
        this.prefixCheckpointAnchors = List.copyOf(copy);
    }

    public KvBufferCacheSettings withPrefixCheckpointAnchors(List<Integer> prefixCheckpointAnchors) {
        setPrefixCheckpointAnchors(prefixCheckpointAnchors);
        return this;
    }

    public int getMaxEntries() {
        return maxEntries;
    }

    public void setMaxEntries(int maxEntries) {
        this.maxEntries = maxEntries;
    }

    public KvBufferCacheSettings withMaxEntries(int maxEntries) {
        this.maxEntries = maxEntries;
        return this;
    }

    public int getBlockSize() {
        return blockSize;
    }

    public void setBlockSize(int blockSize) {
        if (blockSize <= 0){
            throw new IllegalArgumentException("blockSize must be > 0");
        }
        this.blockSize = blockSize;
    }

    public KvBufferCacheSettings withBlockSize(int blockSize) {
        setBlockSize(blockSize);
        return this;
    }

    public int getContextRowsPerPageTarget() {
        return contextRowsPerPageTarget;
    }

    /**
     * Sets the preferred number of adjacent context-token rows per active KV page.
     *
     * <p>Increasing this can reduce attention page fan-out for long contexts. Setting it too high can hurt locality or
     * allocator behavior for some model shapes, so benchmark representative prompts before changing it globally.</p>
     */
    public void setContextRowsPerPageTarget(int contextRowsPerPageTarget) {
        if (contextRowsPerPageTarget <= 0) {
            throw new IllegalArgumentException("contextRowsPerPageTarget must be > 0");
        }
        this.contextRowsPerPageTarget = contextRowsPerPageTarget;
    }

    public KvBufferCacheSettings withContextRowsPerPageTarget(int contextRowsPerPageTarget) {
        setContextRowsPerPageTarget(contextRowsPerPageTarget);
        return this;
    }

    public boolean isDeleteDiskPagesOnClose() {
        return deleteDiskPagesOnClose;
    }

    public void setDeleteDiskPagesOnClose(boolean deleteDiskPagesOnClose) {
        this.deleteDiskPagesOnClose = deleteDiskPagesOnClose;
    }

    public KvBufferCacheSettings withDeleteDiskPagesOnClose(boolean deleteDiskPagesOnClose) {
        setDeleteDiskPagesOnClose(deleteDiskPagesOnClose);
        return this;
    }

    public boolean isDiskPageSweeperEnabled() {
        return diskPageSweeperEnabled;
    }

    public void setDiskPageSweeperEnabled(boolean diskPageSweeperEnabled) {
        this.diskPageSweeperEnabled = diskPageSweeperEnabled;
    }

    public KvBufferCacheSettings withDiskPageSweeperEnabled(boolean diskPageSweeperEnabled) {
        setDiskPageSweeperEnabled(diskPageSweeperEnabled);
        return this;
    }

    public Duration getDiskPageSweepInterval() {
        return diskPageSweepInterval;
    }

    public void setDiskPageSweepInterval(Duration diskPageSweepInterval) {
        if (diskPageSweepInterval == null || diskPageSweepInterval.toMillis() < 1) {
            throw new IllegalArgumentException("diskPageSweepInterval must be at least 1 millisecond");
        }
        this.diskPageSweepInterval = diskPageSweepInterval;
    }

    public KvBufferCacheSettings withDiskPageSweepInterval(Duration diskPageSweepInterval) {
        setDiskPageSweepInterval(diskPageSweepInterval);
        return this;
    }

    public Duration getDiskPageMaxAge() {
        return diskPageMaxAge;
    }

    public void setDiskPageMaxAge(Duration diskPageMaxAge) {
        if (diskPageMaxAge == null || diskPageMaxAge.toMillis() < 1) {
            throw new IllegalArgumentException("diskPageMaxAge must be at least 1 millisecond");
        }
        this.diskPageMaxAge = diskPageMaxAge;
    }

    public KvBufferCacheSettings withDiskPageMaxAge(Duration diskPageMaxAge) {
        setDiskPageMaxAge(diskPageMaxAge);
        return this;
    }
}
