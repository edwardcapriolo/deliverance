package io.teknek.deliverance;

import java.io.File;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class ModelFetcher {
    private static final String HF_TOKEN = "HF_TOKEN";
    private static final String HF_PROP = "huggingface.auth.token";

    private final Path baseDir;
    private final String owner;
    private final String name;

    public ModelFetcher(String owner, String name){
        this.owner = owner;
        this.name = name;
        String home = System.getProperty("user.home");
        baseDir = Path.of(home, ".deliverance");
        if (!Files.exists(baseDir)){
            try {
                Files.createDirectory(baseDir);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }
    }

    public File maybeDownload(){
        Path modelDir = Paths.get(baseDir.toString(), owner + "_" + name );
        if (Files.exists(modelDir)){
            return modelDir.toFile();
        } else {
            throw new RuntimeException("Did not download");
        }
    }
}
