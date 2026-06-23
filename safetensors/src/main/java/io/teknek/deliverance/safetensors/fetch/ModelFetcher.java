package io.teknek.deliverance.safetensors.fetch;

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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

public class ModelFetcher {
    public static final String HF_TOKEN = "HF_TOKEN";
    public static final String HF_PROP = "huggingface.auth.token";
    private static final String FINISHED_MARKER = ".finished";

    public enum FetchPolicy {
        FULL_MODEL,
        TOKENIZER_ONLY
    }

    public record RemoteFileMetadata(String name, long size) {
    }

    protected Path baseDir;
    protected String owner;
    protected String name;
    protected String token;
    protected String modelUriBase = "https://huggingface.co/api/models/";
    protected boolean download = true;

    protected ModelFetcher(){

    }

    public ModelFetcher(String owner, String name){
        this.owner = owner;
        this.name = name;
        String home = System.getProperty("user.home");
        baseDir = Path.of(home, ".deliverance");
        token = System.getenv(HF_TOKEN);
        String tokenProp = System.getProperty(HF_PROP);
        if (tokenProp != null) {
            token = tokenProp;
        }
    }

    public ModelFetcher(String owner, String name, String token){
        this(owner, name);
        this.token = token;
    }

    public File maybeDownload(){
        return maybeDownload(FetchPolicy.FULL_MODEL);
    }

    public File maybeDownload(FetchPolicy fetchPolicy){
        if (!Files.exists(baseDir)){
            try {
                Files.createDirectory(baseDir);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }
        Path modelDir = pathForModel();
        if (isLocallyComplete(fetchPolicy, modelDir)) {
            return modelDir.toFile();
        }
        if (!download) {
            throw new IllegalStateException("Model is not available locally and downloads are disabled: " + modelDir);
        }
        try {
            return maybeDownloadModel(Optional.of(this.owner), this.name,
                    fetchPolicy,
                    Optional.empty(),
                    token != null ? Optional.of(token): Optional.empty(), modelUriBase);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /**
     *
     * @return The path to where the model might exists
     */
    public Path pathForModel(){
        return Paths.get(baseDir.toString(), owner + "_" + name);
    }

    /**
     * Download a model from HuggingFace and return the path to the model directory
     *
     * @param modelOwner The owner of the HF model (if any)
     * @param modelName The name of the HF model
     * @param downloadWeights Include the weights or leave them out
     * @param optionalBranch The branch of the model to download
     * @param optionalAuthHeader The authorization header to use for the request

     * @return The path to the downloaded model directory
     * @throws IOException
     */
    protected File maybeDownloadModel(
            Optional<String> modelOwner,
            String modelName,
            FetchPolicy fetchPolicy,
            Optional<String> optionalBranch,
            Optional<String> optionalAuthHeader,
            String baseUrl
    ) throws IOException {
        Path localModelDir = pathForModel();
        String hfModel = modelOwner.map(mo -> mo + "/" + modelName).orElse(modelName);
        InputStream modelInfoStream = HttpSupport.getResponse(
                baseUrl + hfModel + "/tree/" + optionalBranch.orElse("main"),
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
        List<String> filesToDownload = filesToDownload(allFiles, fetchPolicy == FetchPolicy.FULL_MODEL);
        Map<String, RemoteFileMetadata> metadata = fetchRemoteMetadata(hfModel, filesToDownload, optionalBranch, optionalAuthHeader);
        Files.createDirectories(localModelDir);
        List<String> incompleteFiles = findIncompleteFiles(metadata, localModelDir);
        for (String currFile : incompleteFiles) {
            HttpSupport.downloadFile(
                    hfModel,
                    currFile,
                    optionalBranch,
                    optionalAuthHeader,
                    Optional.empty(),
                    localModelDir.resolve(currFile)
            );
        }
        if (fetchPolicy == FetchPolicy.FULL_MODEL) {
            Files.deleteIfExists(localModelDir.resolve(FINISHED_MARKER));
            Files.createFile(localModelDir.resolve(FINISHED_MARKER));
        }
        return localModelDir.toFile();
    }

    protected Map<String, RemoteFileMetadata> fetchRemoteMetadata(
            String hfModel,
            List<String> filesToDownload,
            Optional<String> optionalBranch,
            Optional<String> optionalAuthHeader
    ) throws IOException {
        Map<String, RemoteFileMetadata> metadata = new LinkedHashMap<>();
        for (String currFile : filesToDownload) {
            long size = HttpSupport.getResponse(
                    "https://huggingface.co/" + hfModel + "/resolve/" + optionalBranch.orElse("main") + "/" + currFile,
                    optionalAuthHeader,
                    Optional.of(Pair.of(0L, 0L))
            ).getRight();
            metadata.put(currFile, new RemoteFileMetadata(currFile, size));
        }
        return metadata;
    }

    protected List<String> findIncompleteFiles(Map<String, RemoteFileMetadata> remoteFiles, Path localModelDir) {
        List<String> incomplete = new ArrayList<>();
        for (RemoteFileMetadata metadata : remoteFiles.values()) {
            if (!isFileComplete(localModelDir.resolve(metadata.name()), metadata)) {
                incomplete.add(metadata.name());
            }
        }
        return incomplete;
    }

    protected boolean isFileComplete(Path localFile, RemoteFileMetadata remoteFile) {
        File file = localFile.toFile();
        return file.exists() && file.length() > 0 && file.length() == remoteFile.size();
    }

    /**
     * Allows locally created model directories (for example quantized outputs) to be used offline
     * without first contacting Hugging Face. The check is intentionally simple: required files must
     * exist and be non-empty for the requested fetch policy.
     */
    protected boolean isLocallyComplete(FetchPolicy fetchPolicy, Path localModelDir) {
        if (!Files.isDirectory(localModelDir)) {
            return false;
        }
        if (Files.exists(localModelDir.resolve(FINISHED_MARKER))) {
            return true;
        }
        if (fetchPolicy == FetchPolicy.TOKENIZER_ONLY) {
            return hasAnyNonEmptyFile(localModelDir,
                    "tokenizer.json",
                    "tokenizer.model",
                    "tokenizer_config.json",
                    "vocab.json",
                    "merges.txt");
        }
        return hasNonEmptyFile(localModelDir, "config.json")
                && hasAnyNonEmptyWeightFile(localModelDir)
                && hasAnyNonEmptyFile(localModelDir,
                "tokenizer.json",
                "tokenizer.model",
                "tokenizer_config.json",
                "vocab.json",
                "merges.txt");
    }

    protected boolean hasAnyNonEmptyWeightFile(Path localModelDir) {
        File[] files = localModelDir.toFile().listFiles();
        if (files == null) {
            return false;
        }
        for (File file : files) {
            String name = file.getName().toLowerCase();
            if (file.isFile() && file.length() > 0 && (name.endsWith(".safetensors") || name.equals("model.safetensors.index.json"))) {
                return true;
            }
        }
        return false;
    }

    protected boolean hasAnyNonEmptyFile(Path localModelDir, String... names) {
        for (String name : names) {
            if (hasNonEmptyFile(localModelDir, name)) {
                return true;
            }
        }
        return false;
    }

    protected boolean hasNonEmptyFile(Path localModelDir, String name) {
        File file = localModelDir.resolve(name).toFile();
        return file.isFile() && file.length() > 0;
    }

    protected List<String> filesToDownload(List<String> allFiles, boolean downloadWeights){
        List<String> filesToDownload = new ArrayList<>();
        boolean hasSafetensor = false;
        for (String currFile : allFiles) {
            String lowerCaseFile = currFile.toLowerCase();
            if ((lowerCaseFile.contains("safetensor") && !lowerCaseFile.contains("consolidated"))
                    || lowerCaseFile.contains("readme")
                    || lowerCaseFile.equals("config.json")
                    || lowerCaseFile.equals("chat_template.jinja")
                    || lowerCaseFile.contains("tokenizer")) {
                if (lowerCaseFile.contains("safetensor")) {
                    hasSafetensor = true;
                }
                if (!downloadWeights && lowerCaseFile.contains("safetensor")) {
                    continue;
                }
                filesToDownload.add(currFile);
            }
        }
        if (!hasSafetensor) {
            throw new RuntimeException("Model is not available in safetensor format");
        }
        return  filesToDownload;
    }

    protected static List<String> parseFileList(String modelInfo) throws IOException {
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

    public String getModelUriBase() {
        return modelUriBase;
    }

    public void setModelUriBase(String modelUriBase) {
        this.modelUriBase = modelUriBase;
    }

    public String getOwner() {
        return owner;
    }

    public String getName() {
        return name;
    }

    public Path getBaseDir() {
        return baseDir;
    }

    public void setBaseDir(Path baseDir) {
        this.baseDir = baseDir;
    }

    public boolean isDownload() {
        return download;
    }

    public void setDownload(boolean download) {
        this.download = download;
    }

    public ModelFetcher withDownload(boolean download) {
        this.download = download;
        return this;
    }
}
