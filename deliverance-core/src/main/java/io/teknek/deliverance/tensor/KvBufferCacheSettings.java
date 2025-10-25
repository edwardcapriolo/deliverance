package io.teknek.deliverance.tensor;

import javax.annotation.Nullable;
import java.io.File;

public class KvBufferCacheSettings {
    private final Boolean useTensorCache;
    private final File workingDirectory;
    private final TensorCache dedicatedCache;

    /**
     * Use the tensor cache shared with model
     */
    public KvBufferCacheSettings(boolean ephemeral) {
        this.useTensorCache = ephemeral;
        this.dedicatedCache = null;
        this.workingDirectory = null;
    }

    /**
     * Use a dedicated tensor cache
     */
    public KvBufferCacheSettings(TensorCache cache){
        this.useTensorCache = true;
        this.dedicatedCache = cache;
        this.workingDirectory = null;
    }

    public KvBufferCacheSettings(File workingDirectory) {
        this.workingDirectory = workingDirectory;
        this.useTensorCache = false;
        this.dedicatedCache = null;
    }

    public boolean isEphemeral() {
        return Boolean.TRUE.equals(useTensorCache);
    }

    @Nullable
    public File getWorkingDirectory() {
        return this.workingDirectory;
    }

    @Nullable
    public TensorCache getDedicatedCache(){
        return this.dedicatedCache;
    }
}
