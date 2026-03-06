package io.teknek.deliverance.model;

import com.codahale.metrics.MetricRegistry;
import com.fasterxml.jackson.databind.JsonNode;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.model.bert.BertModelType;
import io.teknek.deliverance.model.gemma2.Gemma2ModelType;
import io.teknek.deliverance.model.llama.LlamaModelType;
import io.teknek.deliverance.model.qwen2.Qwen2ModelType;
import io.teknek.deliverance.model.qwen2.Qwen2TokenizerRenderer;
import io.teknek.deliverance.safetensors.*;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorCache;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tokenizer.Tokenizer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import java.io.File;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.nio.file.Path;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

import static io.teknek.deliverance.JsonUtils.om;

public class ModelSupport {
    private static final Logger LOGGER = LoggerFactory.getLogger(ModelSupport.class);
    private static final ConcurrentMap<String,ModelType> registry = new ConcurrentHashMap<String, ModelType>();

    static {
        registry.putIfAbsent("BERT", new BertModelType());
        registry.putIfAbsent("LLAMA", new LlamaModelType());
        registry.putIfAbsent("QWEN2", new Qwen2ModelType());
        registry.putIfAbsent("GEMMA2", new Gemma2ModelType());
    }

    public static void addModel(String modelName, ModelType t){
        registry.putIfAbsent(modelName, t);
    }

    public static @Nonnull ModelType getModelType(String modelType) {
        LOGGER.info("Seeking a model of type {} from the registry. ", modelType);
        ModelType found = registry.get(modelType);
        if (found == null){
            throw new IllegalArgumentException(modelType + " not found in registry");
        }
        return found;
    }

    public static ModelType detectModel(File configFile) {
        JsonNode rootNode;
        try {
            rootNode = JsonUtils.om.readTree(configFile);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        if (!rootNode.has("model_type")) {
            throw new IllegalArgumentException("Config missing model_type field.");
        }
        return ModelSupport.getModelType(rootNode.get("model_type").textValue().toUpperCase());
    }

    public static AbstractModel loadModel(File model, DType workingMemoryType, DType workingQuantizationType,
                                          ConfigurableTensorProvider configurableTensorProvider,
                                          MetricRegistry metricRegistry, TensorCache tensorCache,
                                          KvBufferCacheSettings kvBufferCacheSettings,
                                          ModelFetcher fetcher){
        //not all llama models use same tokenizer. detecting from config.json might be an option
        TokenRenderer tr;
        if (fetcher.getName().startsWith("Llama-3.1-8B-Instruct")
                || fetcher.getName().startsWith("Llama-3.2-3B-Instruct")){
            tr = new TokenizerRenderer();
        } else if (fetcher.getName().startsWith("Qwen2.5-0.5B-Instruct")
                || fetcher.getName().startsWith("Qwen2.5-3B-Instruct")) {
          tr = new Qwen2TokenizerRenderer();
        } else {
            tr = new NoOpTokenizerRenderer();
        }
        File configFile = new File(model, "config.json");
        if (!configFile.exists()){
            throw new RuntimeException("Expecting to find config file " + configFile);
        }
        ModelType modelType = detectModel(configFile);
        try {
            Config config = om.readValue(configFile, modelType.getConfigClass());
            Tokenizer tokenizer = modelType.getTokenizerClass().getConstructor(Path.class).newInstance(model.toPath());
            WeightLoader wl = new DefaultWeightLoader(model);

            Constructor<? extends AbstractModel> cons = modelType.getModelClass().getConstructor(AbstractModel.InferenceType.class, Config.class,
                    WeightLoader.class, Tokenizer.class, DType.class, DType.class, Optional.class,
                    ConfigurableTensorProvider.class, MetricRegistry.class, TensorCache.class,
                    KvBufferCacheSettings.class, TokenRenderer.class);

            return cons.newInstance(AbstractModel.InferenceType.FULL_GENERATION, config, wl, tokenizer,
                    workingMemoryType, workingQuantizationType, Optional.empty(), configurableTensorProvider,
                    metricRegistry, tensorCache, kvBufferCacheSettings, tr);
        } catch (IOException | NoSuchMethodException | InvocationTargetException | InstantiationException | IllegalAccessException e) {
            throw new RuntimeException(e);
        }
    }

    public static AbstractModel  loadEmbeddingModel(File model, DType workingMemoryType, DType workingQuantizationType,
                                                    ConfigurableTensorProvider configurableTensorProvider,
                                                    MetricRegistry metricRegistry, TensorCache tensorCache,
                                                    KvBufferCacheSettings kvBufferCacheSettings) {
     return load(AbstractModel.InferenceType.FULL_EMBEDDING,model, workingMemoryType, workingQuantizationType, configurableTensorProvider, metricRegistry, tensorCache,kvBufferCacheSettings);

    }
    protected static AbstractModel load(AbstractModel.InferenceType infType, File model, DType workingMemoryType, DType workingQuantizationType,
                                 ConfigurableTensorProvider configurableTensorProvider,
                                 MetricRegistry metricRegistry, TensorCache tensorCache,
                                 KvBufferCacheSettings kvBufferCacheSettings) {
        File configFile = new File(model, "config.json");
        if (!configFile.exists()){
            throw new RuntimeException("Expecting to find config file " + configFile);
        }
        ModelType modelType = detectModel(configFile);

        try {
            Config config = om.readValue(configFile, modelType.getConfigClass());
            Tokenizer tokenizer = modelType.getTokenizerClass().getConstructor(Path.class).newInstance(model.toPath());

            WeightLoader wl = new DefaultWeightLoader(model);

            Constructor<? extends AbstractModel> cons = modelType.getModelClass().getConstructor(AbstractModel.InferenceType.class, Config.class,
                    WeightLoader.class, Tokenizer.class, DType.class, DType.class, Optional.class,
                    ConfigurableTensorProvider.class, MetricRegistry.class, TensorCache.class,
                    KvBufferCacheSettings.class, TokenRenderer.class);

            return cons.newInstance(infType, config, wl, tokenizer,
                    workingMemoryType, workingQuantizationType, Optional.empty(), configurableTensorProvider,
                    metricRegistry, tensorCache, kvBufferCacheSettings, new NoOpTokenizerRenderer());
        } catch (IOException | NoSuchMethodException | InvocationTargetException | InstantiationException | IllegalAccessException e) {
            throw new RuntimeException(e);
        }
    }
/*
    public static AbstractModel loadEmbeddingModel(File model, DType workingMemoryType, DType workingQuantizationType) {
        AbstractModel embed = loadModel(AbstractModel.InferenceType.FULL_EMBEDDING, ConfigurableTensorProvider provider
                model, workingMemoryType, workingQuantizationType )
        /*
        return loadModel(
                AbstractModel.InferenceType.FULL_EMBEDDING,
                model,
                null,
                workingMemoryType,
                workingQuantizationType,
                Optional.empty(),
                Optional.empty(),
                Optional.empty()
        );


    }
*/
}
