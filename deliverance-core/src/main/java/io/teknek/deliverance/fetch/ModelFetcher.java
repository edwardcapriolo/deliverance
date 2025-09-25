package io.teknek.deliverance.fetch;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

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

public class ModelFetcher {
    private static final String HF_TOKEN = "HF_TOKEN";
    private static final String HF_PROP = "huggingface.auth.token";
    private static final String FINISHED_MARKER = ".finished";

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
            try {
                return maybeDownloadModel(baseDir.toString(), Optional.of(this.owner), this.name, true, Optional.empty(), Optional.empty());
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
     * @param downloadWeights Include the weights or leave them out
     * @param optionalBranch The branch of the model to download
     * @param optionalAuthHeader The authorization header to use for the request

     * @return The path to the downloaded model directory
     * @throws IOException
     */
    protected static File maybeDownloadModel(
            String modelDir,
            Optional<String> modelOwner,
            String modelName,
            boolean downloadWeights,
            Optional<String> optionalBranch,
            Optional<String> optionalAuthHeader
    ) throws IOException {
        Path localModelDir = constructLocalModelPath(modelDir, modelOwner.orElse("na"), modelName);
        if (Files.exists(localModelDir.resolve(FINISHED_MARKER))) {
            return localModelDir.toFile();
        }
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
        boolean hasSafetensor = false;
        for (String currFile : allFiles) {
            String f = currFile.toLowerCase();
            if ((f.contains("safetensor") && !f.contains("consolidated"))
                    || f.contains("readme")
                    || f.equals("config.json")
                    || f.contains("tokenizer")) {
                if (f.contains("safetensor")) {
                    hasSafetensor = true;
                }
                if (!downloadWeights && f.contains("safetensor")) {
                    continue;
                }
                tensorFiles.add(currFile);
            }
        }
        if (!hasSafetensor) {
            throw new IOException("Model is not available in safetensor format");
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
        Files.createFile(localModelDir.resolve(FINISHED_MARKER));
        return localModelDir.toFile();
    }

    private static List<String> parseFileList(String modelInfo) throws IOException {
        List<String> fileList = new ArrayList<>();

        ObjectMapper objectMapper = new ObjectMapper();
        JsonNode siblingsNode = objectMapper.readTree(modelInfo);
        if (siblingsNode.isArray()) {
            for (JsonNode siblingNode : siblingsNode) {
                String rFilename = siblingNode.path("path").asText();
                fileList.add(rFilename);
            }
        }

        return fileList;
    }

    private static File maybeDownloadModel(String modelDir, String fullModelName) throws IOException {
        String[] parts = fullModelName.split("/");
        if (parts.length == 0 || parts.length > 2) {
            throw new IllegalArgumentException("Model must be in the form owner/name");
        }

        String owner;
        String name;

        if (parts.length == 1) {
            owner = null;
            name = fullModelName;
        } else {
            owner = parts[0];
            name = parts[1];
        }

        return maybeDownloadModel(
                modelDir,
                Optional.ofNullable(owner),
                name,
                true,
                Optional.empty(),
                Optional.empty()
        );
    }

    public static Path constructLocalModelPath(String modelDir, String owner, String modelName) {
        return Paths.get(modelDir, owner + "_" + modelName);
    }

}
