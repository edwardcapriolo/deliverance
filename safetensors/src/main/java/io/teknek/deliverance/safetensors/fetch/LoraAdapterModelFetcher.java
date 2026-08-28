package io.teknek.deliverance.safetensors.fetch;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Fetches a HuggingFace PEFT-format LoRA adapter repo (an {@code adapter_config.json} plus an
 * {@code adapter_model.safetensors}), following the same subclassing pattern as {@code
 * grace.TokenizerModelFetcher}: {@link ModelFetcher#filesToDownload} is already {@code
 * protected}, so no base-class changes are needed to select a different file set for a
 * different kind of HuggingFace repo.
 */
public class LoraAdapterModelFetcher extends ModelFetcher {
    private static final String FINISHED_MARKER = ".finished";
    private final String subfolder;

    public LoraAdapterModelFetcher(String owner, String name) {
        super(owner, name);
        this.subfolder = null;
    }

    public LoraAdapterModelFetcher(String owner, String name, String token) {
        super(owner, name, token);
        this.subfolder = null;
    }

    public LoraAdapterModelFetcher(String owner, String name, String subfolder, boolean subfolderAdapter) {
        super(owner, name);
        this.subfolder = normalizeSubfolder(subfolderAdapter ? subfolder : null);
    }

    public LoraAdapterModelFetcher(String owner, String name, String subfolder, String token) {
        super(owner, name, token);
        this.subfolder = normalizeSubfolder(subfolder);
    }

    @Override
    public Path pathForModel() {
        if (subfolder == null) {
            return super.pathForModel();
        }
        return Paths.get(baseDir.toString(), owner + "_" + name + "_" + subfolder.replace('/', '_'));
    }

    @Override
    protected File maybeDownloadModel(Optional<String> modelOwner, String modelName, FetchPolicy fetchPolicy,
            Optional<String> optionalBranch, Optional<String> optionalAuthHeader, String baseUrl) throws IOException {
        if (subfolder == null) {
            return super.maybeDownloadModel(modelOwner, modelName, fetchPolicy, optionalBranch, optionalAuthHeader,
                    baseUrl);
        }
        Path localModelDir = pathForModel();
        String hfModel = modelOwner.map(mo -> mo + "/" + modelName).orElse(modelName);
        String modelInfo = HttpSupport.readInputStream(HttpSupport.getResponse(
                baseUrl + hfModel + "/tree/" + optionalBranch.orElse("main") + "?recursive=1",
                optionalAuthHeader,
                Optional.empty()).getLeft());
        List<String> filesToDownload = filesToDownload(parseFileList(modelInfo), true);
        Map<String, RemoteFileMetadata> metadata = new LinkedHashMap<>();
        for (String remoteFile : filesToDownload) {
            String localName = stripSubfolder(remoteFile);
            long size = HttpSupport.getResponse(
                    "https://huggingface.co/" + hfModel + "/resolve/" + optionalBranch.orElse("main") + "/" + remoteFile,
                    optionalAuthHeader,
                    Optional.of(Pair.of(0L, 0L))).getRight();
            metadata.put(localName, new RemoteFileMetadata(localName, size));
        }
        Files.createDirectories(localModelDir);
        for (String remoteFile : filesToDownload) {
            String localName = stripSubfolder(remoteFile);
            RemoteFileMetadata remoteMetadata = metadata.get(localName);
            if (!isFileComplete(localModelDir.resolve(localName), remoteMetadata)) {
                HttpSupport.downloadFile(hfModel, remoteFile, optionalBranch, optionalAuthHeader, Optional.empty(),
                        localModelDir.resolve(localName));
            }
        }
        Files.deleteIfExists(localModelDir.resolve(FINISHED_MARKER));
        Files.createFile(localModelDir.resolve(FINISHED_MARKER));
        return localModelDir.toFile();
    }

    @Override
    protected boolean isLocallyComplete(FetchPolicy fetchPolicy, Path localModelDir) {
        if (subfolder == null) {
            return super.isLocallyComplete(fetchPolicy, localModelDir);
        }
        return Files.isDirectory(localModelDir)
                && hasNonEmptyFile(localModelDir, "adapter_config.json")
                && hasNonEmptyFile(localModelDir, "adapter_model.safetensors");
    }

    @Override
    protected List<String> filesToDownload(List<String> allFiles, boolean downloadWeights) {
        List<String> files = new ArrayList<>();
        for (String currFile : allFiles) {
            String f = currFile.toLowerCase();
            if (subfolder != null && !f.startsWith(subfolder + "/")) {
                continue;
            }
            String localName = subfolder == null ? f : f.substring(subfolder.length() + 1);
            if (localName.contains("safetensor") || localName.equals("adapter_config.json")
                    || localName.contains("readme")) {
                files.add(currFile);
            }
        }
        return files;
    }

    private String stripSubfolder(String remoteFile) {
        return remoteFile.substring(subfolder.length() + 1);
    }

    private static String normalizeSubfolder(String value) {
        if (value == null || value.isBlank()) {
            return null;
        }
        String normalized = value;
        while (normalized.startsWith("/")) {
            normalized = normalized.substring(1);
        }
        while (normalized.endsWith("/")) {
            normalized = normalized.substring(0, normalized.length() - 1);
        }
        return normalized.isBlank() ? null : normalized;
    }
}
