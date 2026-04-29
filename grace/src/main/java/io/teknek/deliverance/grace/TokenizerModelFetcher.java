package io.teknek.deliverance.grace;

import io.teknek.deliverance.safetensors.fetch.HttpSupport;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

public class TokenizerModelFetcher extends ModelFetcher {

    public TokenizerModelFetcher(String owner, String name) {
        super(owner, name);
        baseDir = baseDir.resolve("tokenizers");
    }

    public TokenizerModelFetcher(String owner, String name, String token){
        super(owner, name, token);
        baseDir = baseDir.resolve("tokenizers");
        baseDir.toFile().mkdirs();
    }

    public File maybeDownload(){
        Path modelDir = Paths.get(baseDir.toString(), owner + "_" + name);
        if (Files.exists(modelDir)){
            return modelDir.toFile();
        } else {
            try {
                return maybeDownloadModel(Optional.of(this.owner), this.name,
                        true, Optional.empty(),
                        token != null ? Optional.of(token): Optional.empty(), modelUriBase);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }
    }

    @Override
    protected List<String> filesToDownload(List<String> allFiles, boolean downloadWeights) {
        List<String> tensorFiles = new ArrayList<>();
        for (String currFile : allFiles) {
            String f = currFile.toLowerCase();
            if (f.contains("tokenizer") || f.contains("merges") || f.contains("vocab") || f.equals("config.json") ) {
                tensorFiles.add(currFile);
            }
        }
        return tensorFiles;
    }

}
