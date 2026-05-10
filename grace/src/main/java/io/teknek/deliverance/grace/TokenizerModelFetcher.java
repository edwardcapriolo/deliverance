package io.teknek.deliverance.grace;

import io.teknek.deliverance.safetensors.fetch.ModelFetcher;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class TokenizerModelFetcher extends ModelFetcher {

    public TokenizerModelFetcher(String owner, String name) {
        super(owner, name);
    }

    public TokenizerModelFetcher(String owner, String name, String token){
        super(owner, name, token);
    }

    public File maybeDownload(){
        return super.maybeDownload(ModelFetcher.FetchPolicy.TOKENIZER_ONLY);
    }

    @Override
    protected List<String> filesToDownload(List<String> allFiles, boolean downloadWeights) {
        List<String> tensorFiles = new ArrayList<>();
        for (String currFile : allFiles) {
            String f = currFile.toLowerCase();
            if (f.contains("tokenizer") || f.contains("merges") || f.contains("vocab") || f.equals("config.json")
                    || f.equals("chat_template.jinja")) {
                tensorFiles.add(currFile);
            }
        }
        return tensorFiles;
    }

}
