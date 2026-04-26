package io.teknek.deliverance.tensor;

import javax.annotation.Nullable;
import java.io.File;

public class KvBufferCacheSettings {
    private final Boolean useTensorAllocator;
    private final File workingDirectory;
    private final TensorAllocator dedicatedCache;
    private int maxPrefixTokensPerPrompt = 512;
    private int maxEntries = 10_000;
    private int blockSize = 8;

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
}
