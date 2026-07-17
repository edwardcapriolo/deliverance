package io.teknek.deliverance.safetensors.fetch;

import java.util.ArrayList;
import java.util.List;

/**
 * Fetches a HuggingFace PEFT-format LoRA adapter repo (an {@code adapter_config.json} plus an
 * {@code adapter_model.safetensors}), following the same subclassing pattern as {@code
 * grace.TokenizerModelFetcher}: {@link ModelFetcher#filesToDownload} is already {@code
 * protected}, so no base-class changes are needed to select a different file set for a
 * different kind of HuggingFace repo.
 */
public class LoraAdapterModelFetcher extends ModelFetcher {

    public LoraAdapterModelFetcher(String owner, String name) {
        super(owner, name);
    }

    public LoraAdapterModelFetcher(String owner, String name, String token) {
        super(owner, name, token);
    }

    @Override
    protected List<String> filesToDownload(List<String> allFiles, boolean downloadWeights) {
        List<String> files = new ArrayList<>();
        for (String currFile : allFiles) {
            String f = currFile.toLowerCase();
            if (f.contains("safetensor") || f.equals("adapter_config.json") || f.contains("readme")) {
                files.add(currFile);
            }
        }
        return files;
    }
}
