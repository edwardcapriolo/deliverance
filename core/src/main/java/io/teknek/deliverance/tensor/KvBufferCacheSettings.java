package io.teknek.deliverance.tensor;

import javax.annotation.Nullable;
import java.io.File;

public class KvBufferCacheSettings {
    private final Boolean useTensorAllocator;
    private final File workingDirectory;
    private final TensorAllocator dedicatedCache;

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
}
