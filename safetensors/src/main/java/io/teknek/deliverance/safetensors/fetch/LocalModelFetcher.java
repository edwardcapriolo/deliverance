package io.teknek.deliverance.safetensors.fetch;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;

/** ModelFetcher for an already-downloaded local model directory. */
public final class LocalModelFetcher extends ModelFetcher {
    private final Path modelDirectory;

    public LocalModelFetcher(Path modelDirectory) {
        super();
        this.modelDirectory = modelDirectory.toAbsolutePath().normalize();
        if (!Files.isDirectory(this.modelDirectory)) {
            throw new IllegalArgumentException("Model directory does not exist: " + this.modelDirectory);
        }
        this.owner = "local";
        this.name = this.modelDirectory.getFileName().toString();
        this.baseDir = this.modelDirectory.getParent();
        this.download = false;
    }

    @Override
    public File maybeDownload() {
        return modelDirectory.toFile();
    }

    @Override
    public Path pathForModel() {
        return modelDirectory;
    }
}
