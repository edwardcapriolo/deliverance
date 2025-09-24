package io.teknek.deliverance.model;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.model.llama.LlamaModelType;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.safetensors.SafeTensorSupport;
import io.teknek.deliverance.tokenizer.Tokenizer;

import java.io.File;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class ModelSupport {

    private static final Map<String,ModelType> registry = new ConcurrentHashMap<String, ModelType>();

    static {
        registry.putIfAbsent("LLAMA", new LlamaModelType());
    }

    public static ModelType getModelType(String modelType) {
        return registry.get(modelType);
    }

    AbstractModel loadModel(File model, DType workingMemoryType, DType workingQuantizationType){
        File config = new File(model, "config.json");
        if (!config.exists()){
            throw new RuntimeException("expecting to find config " + config);
        }
        ModelType modelType = SafeTensorSupport.detectModel(config);
        return null;
    }

    public interface ModelType {
        Class<? extends AbstractModel> getModelClass();

        Class<? extends Config> getConfigClass();

        Class<? extends Tokenizer> getTokenizerClass();
    }

}
