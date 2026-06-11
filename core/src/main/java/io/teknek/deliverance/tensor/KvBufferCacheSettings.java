package io.teknek.deliverance.tensor;

import javax.annotation.Nullable;
import java.io.File;
import java.time.Duration;

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
    /** the maximum size of the cache before evictions happen **/
    private int maxEntries = 10_000;
    /**
    The block size of the kvcache. Cache hits will only happen at block boundaries, smaller blockize uses more memory
     */
    private int blockSize = 32;

    public enum KvFormat {
        BF16,
        F32,
        QUANTIZED_INT8
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
