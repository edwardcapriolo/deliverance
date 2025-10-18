package io.teknek.deliverance.model;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.model.llama.LlamaModelType;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.safetensors.DefaultWeightLoader;
import io.teknek.deliverance.safetensors.SafeTensorSupport;
import io.teknek.deliverance.safetensors.WeightLoader;
import io.teknek.deliverance.tensor.TensorCache;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tokenizer.Tokenizer;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.nio.file.Path;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

import static io.teknek.deliverance.JsonUtils.om;

public class ModelSupport {

    private static final Map<String,ModelType> registry = new ConcurrentHashMap<String, ModelType>();

    static {
        registry.putIfAbsent("LLAMA", new LlamaModelType());
    }

    public static ModelType getModelType(String modelType) {
        return registry.get(modelType);
    }

    public static AbstractModel loadModel(File model, DType workingMemoryType, DType workingQuantizationType,
                                          ConfigurableTensorProvider configurableTensorProvider, MetricRegistry metricRegistry, TensorCache tensorCache){
        File configFile = new File(model, "config.json");
        if (!configFile.exists()){
            throw new RuntimeException("expecting to find config " + configFile);
        }
        ModelType modelType = SafeTensorSupport.detectModel(configFile);
        try {
            Config config = om.readValue(configFile, modelType.getConfigClass());
            config.setWorkingDirectory(null);
            Tokenizer tokenizer = modelType.getTokenizerClass().getConstructor(Path.class).newInstance(model.toPath());
            WeightLoader wl = new DefaultWeightLoader(model);

            Constructor<? extends AbstractModel> cons = modelType.getModelClass().getConstructor(AbstractModel.InferenceType.class, Config.class,
                    WeightLoader.class, Tokenizer.class, DType.class, DType.class, Optional.class, ConfigurableTensorProvider.class, MetricRegistry.class, TensorCache.class);

            return cons.newInstance(AbstractModel.InferenceType.FULL_GENERATION, config, wl, tokenizer,
                    workingMemoryType, workingQuantizationType, Optional.empty(), configurableTensorProvider, metricRegistry, tensorCache);
        } catch (IOException | NoSuchMethodException | InvocationTargetException | InstantiationException | IllegalAccessException e) {
            throw new RuntimeException(e);
        }
    }

}
