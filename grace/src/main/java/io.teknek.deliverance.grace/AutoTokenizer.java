package io.teknek.deliverance.grace;

import io.teknek.deliverance.safetensors.fetch.ModelFetcher;

import java.io.File;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class AutoTokenizer {

    private static ConcurrentHashMap<String, Class<PreTrainedTokenizerBase>> registry = new ConcurrentHashMap<>();
    static {
        registry.put("BERT", null);
    }
    PreTrainedTokenizer fromPretrained(OwnerNameOrPath ownerNameOrPath,
                                       java.util.Optional<String> tokenizerType,
                                       java.util.Optional<java.util.List<Object>> inputs,
                                       Map<String, ?> tokenizerInitArgs){
        File path = null;
        if (ownerNameOrPath.ownerName != null){
            ModelFetcher mf = new ModelFetcher(ownerNameOrPath.ownerName.owner, ownerNameOrPath.ownerName.name);
            path = mf.maybeDownload();
        }
        Class<PreTrainedTokenizerBase> clazz =  null;
        if (tokenizerType.isPresent()){
            clazz = registry.get(tokenizerType.get());
            if (clazz == null){
                throw new RuntimeException("The specified tokenizer_type is not found: " + tokenizerType.get()
                        + " in registry: " + registry);
            }
        }
        return null;
        /*
        File configFile = new File(path, "config.json");
        if (!configFile.exists()){
            throw new RuntimeException("Expecting to find config file " + configFile);
        }
        ModelType modelType = detectModel(configFile);
        */
    }

    static class OwnerName{
        String owner;
        String name;
    }
    static class ModelPath {
        Path path;
    }
    static class OwnerNameOrPath{
        private final OwnerName ownerName;
        private final ModelPath modelPath;
        public OwnerNameOrPath(OwnerName ownerName){
            this.ownerName = ownerName;
            this.modelPath = null;
        }

        public OwnerNameOrPath(ModelPath modelPath){
            this.modelPath = modelPath;
            this.ownerName = null;
        }
    }
}
