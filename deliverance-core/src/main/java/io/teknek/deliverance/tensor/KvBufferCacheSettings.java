package io.teknek.deliverance.tensor;

import java.io.File;

public class KvBufferCacheSettings {
    private Boolean useTensorCache;
    private File workingDirectory;

    /* ephemeral = true ? use tensorCache */
    public KvBufferCacheSettings(boolean ephemeral) {
        this.useTensorCache = ephemeral;
    }

    public KvBufferCacheSettings(File workingDirectory) {
        this.workingDirectory = workingDirectory;
    }

    public boolean isEphemeral() {
        return Boolean.TRUE.equals(useTensorCache);
    }

    public File getWorkingDirectory() {
        return this.workingDirectory;
    }
}
