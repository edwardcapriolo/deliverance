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
        super.owner = owner;
        super.name = name;

        String home = System.getProperty("user.home");
        baseDir = Path.of(home, ".deliverance_tok");
        token = System.getenv(HF_TOKEN);

        String tokenProp = System.getProperty(HF_PROP);
        if (tokenProp != null) {
            token = tokenProp;
        }
        if (!Files.exists(baseDir)){
            try {
                Files.createDirectory(baseDir);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }
    }

    public TokenizerModelFetcher(String owner, String name, String token){
        this(owner, name);
    }

    public File maybeDownload(){
        Path modelDir = Paths.get(baseDir.toString(), owner + "_" + name);
        if (Files.exists(modelDir)){
            return modelDir.toFile();
        } else {
            try {
                return this.maybeDownloadModel(baseDir.toString(), Optional.of(this.owner), this.name,
                        Optional.empty(), token != null ? Optional.of(token): Optional.empty());
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }
    }

    /**
     * Download a model from HuggingFace and return the path to the model directory
     *
     * @param modelDir The directory to save the model to
     * @param modelOwner The owner of the HF model (if any)
     * @param modelName The name of the HF model
     * @param optionalBranch The branch of the model to download
     * @param optionalAuthHeader The authorization header to use for the request

     * @return The path to the downloaded model directory
     * @throws IOException
     */
    protected static File maybeDownloadModel(
            String modelDir,
            Optional<String> modelOwner,
            String modelName,
            Optional<String> optionalBranch,
            Optional<String> optionalAuthHeader
    ) throws IOException {
        Path localModelDir = constructLocalModelPath(modelDir, modelOwner.orElse("na"), modelName);
        String hfModel = modelOwner.map(mo -> mo + "/" + modelName).orElse(modelName);
        InputStream modelInfoStream = HttpSupport.getResponse(
                "https://huggingface.co/api/models/" + hfModel + "/tree/" + optionalBranch.orElse("main"),
                optionalAuthHeader,
                Optional.empty()
        ).getLeft();
        String modelInfo = HttpSupport.readInputStream(modelInfoStream);
        if (modelInfo == null) {
            throw new IOException("No valid model found or trying to access a restricted model (please include correct access token)");
        }
        List<String> allFiles = parseFileList(modelInfo);
        if (allFiles.isEmpty()) {
            throw new IOException("No valid model found");
        }
        List<String> tensorFiles = new ArrayList<>();

        for (String currFile : allFiles) {
            String f = currFile.toLowerCase();
            //this is only what is needed for qwen2 not authoritative
            if (f.contains("tokenizer") || f.contains("merges") || f.contains("vocab") || f.equals("config.json") ) {

                /*
                tokenizer_config.json: 7.30kB [00:00, 16.3MB/s]
                vocab.json: 2.78MB [00:00, 114MB/s]
                merges.txt: 1.67MB [00:00, 132MB/s]
                tokenizer.json: 7.03MB [00:00, 161MB/s]
                 */
                tensorFiles.add(currFile);
            }
        }

        Files.createDirectories(localModelDir);
        for (String currFile : tensorFiles) {
            HttpSupport.downloadFile(
                    hfModel,
                    currFile,
                    optionalBranch,
                    optionalAuthHeader,
                    Optional.empty(),
                    localModelDir.resolve(currFile)
            );
        }
        return localModelDir.toFile();
    }

}
