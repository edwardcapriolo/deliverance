package io.teknek.deliverance.model;

import com.codahale.metrics.MetricRegistry;
import com.fasterxml.jackson.databind.JsonNode;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.grace.AutoTokenizer;
import io.teknek.deliverance.grace.PreTrainedTokenizer;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.bert.BertModelType;
import io.teknek.deliverance.model.gemma2.Gemma2ModelType;
import io.teknek.deliverance.model.gemma4.Gemma4ModelType;
import io.teknek.deliverance.model.gemma3.Gemma3ModelType;
import io.teknek.deliverance.model.gpt2.Gpt2ModelType;
import io.teknek.deliverance.model.llama.LlamaModelType;
import io.teknek.deliverance.model.mistral.MistralModelType;
import io.teknek.deliverance.model.mixtral.MixtralModelType;
import io.teknek.deliverance.model.qwen2.Qwen2ModelType;
import io.teknek.deliverance.model.tensorparallel.StaticTensorParallelContext;
import io.teknek.deliverance.model.tensorparallel.SingleRankTensorParallelCollectives;
import io.teknek.deliverance.model.tensorparallel.TensorParallelCollectives;
import io.teknek.deliverance.model.tensorparallel.TensorParallelContext;
import io.teknek.deliverance.safetensors.*;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.toolcallparser.DefaultToolCallParser;
import io.teknek.deliverance.toolcallparser.ToolCallParser;
import jdk.incubator.vector.FloatVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import java.io.File;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.function.Function;

import static io.teknek.deliverance.JsonUtils.om;

public class ModelSupport {
    private static final Logger LOGGER = LoggerFactory.getLogger(ModelSupport.class);
    private static final ConcurrentMap<String,ModelType> registry = new ConcurrentHashMap<String, ModelType>();

    static {
        registry.putIfAbsent("BERT", new BertModelType());
        registry.putIfAbsent("LLAMA", new LlamaModelType());
        registry.putIfAbsent("QWEN2", new Qwen2ModelType());
        registry.putIfAbsent("GEMMA2", new Gemma2ModelType());
        registry.putIfAbsent("GEMMA4", new Gemma4ModelType());
        registry.putIfAbsent("GEMMA3_TEXT", new Gemma3ModelType());
        registry.putIfAbsent("MISTRAL", new MistralModelType());
        registry.putIfAbsent("GPT2", new Gpt2ModelType());
        registry.putIfAbsent("MIXTRAL", new MixtralModelType());
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
                                            MetricRegistry metricRegistry,    TensorAllocator arrayQueueTensorAllocator,
                                            KvBufferCacheSettings kvBufferCacheSettings,
                                            ModelFetcher fetcher, ToolCallParser toolCallParser,
                                            WrappedForkJoinPool pool) {
        return loadModel(model, workingMemoryType, workingQuantizationType, configurableTensorProvider, metricRegistry,
                arrayQueueTensorAllocator, kvBufferCacheSettings, fetcher, toolCallParser, pool,
                new StaticTensorParallelContext(0, 1), new SingleRankTensorParallelCollectives());
    }

    public static AbstractModel loadModel(File model, DType workingMemoryType, DType workingQuantizationType,
                                            ConfigurableTensorProvider configurableTensorProvider,
                                            MetricRegistry metricRegistry, TensorAllocator arrayQueueTensorAllocator,
                                            KvBufferCacheSettings kvBufferCacheSettings,
                                            ModelFetcher fetcher, ToolCallParser toolCallParser,
                                            WrappedForkJoinPool pool,
                                            TensorParallelContext tensorParallelContext,
                                            TensorParallelCollectives tensorParallelCollectives) {
        LOGGER.info("Machine Vector Spec: {} Byte Order: {}", FloatVector.SPECIES_PREFERRED.vectorBitSize(), ByteOrder.nativeOrder().toString());
        File configFile = new File(model, "config.json");
        if (!configFile.exists()){
            throw new RuntimeException("Expecting to find config file " + configFile);
        }
        ModelType modelType = detectModel(configFile);
        try {
            Config config = om.readValue(configFile, modelType.getConfigClass());
            PreTrainedTokenizer tokenizer = AutoTokenizer.fromPretrained(model.toPath());
            WeightLoader wl = new DefaultWeightLoader(model);

            Constructor<? extends AbstractModel> cons = modelType.getModelClass().getConstructor(AbstractModel.InferenceType.class, Config.class,
                    WeightLoader.class, PreTrainedTokenizer.class, DType.class, DType.class, Optional.class,
                    ConfigurableTensorProvider.class, MetricRegistry.class, TensorAllocator.class,
                    KvBufferCacheSettings.class, ToolCallParser.class, WrappedForkJoinPool.class,
                    TensorParallelContext.class, TensorParallelCollectives.class);

            AbstractModel am = cons.newInstance(AbstractModel.InferenceType.FULL_GENERATION, config, wl, tokenizer,
                    workingMemoryType, workingQuantizationType, Optional.empty(), configurableTensorProvider,
                    metricRegistry, arrayQueueTensorAllocator, kvBufferCacheSettings, toolCallParser, pool,
                    tensorParallelContext, tensorParallelCollectives);
            return am;
        } catch (IOException | NoSuchMethodException | InvocationTargetException | InstantiationException | IllegalAccessException e) {
            throw new RuntimeException(e);
        }

    }

    //Note complete copy of above only different enuum
    public static AbstractModel loadClassificationModel(File model, DType workingMemoryType, DType workingQuantizationType,
                                           ConfigurableTensorProvider configurableTensorProvider,
                                           MetricRegistry metricRegistry, TensorAllocator arrayQueueTensorAllocator,
                                           KvBufferCacheSettings kvBufferCacheSettings,
                                           ModelFetcher fetcher, ToolCallParser toolCallParser,
                                           WrappedForkJoinPool pool) {
        LOGGER.info("Machine Vector Spec: {} Byte Order: {}", FloatVector.SPECIES_PREFERRED.vectorBitSize(), ByteOrder.nativeOrder().toString());
        File configFile = new File(model, "config.json");
        if (!configFile.exists()){
            throw new RuntimeException("Expecting to find config file " + configFile);
        }
        ModelType modelType = detectModel(configFile);
        try {
            Config config = om.readValue(configFile, modelType.getConfigClass());
            PreTrainedTokenizer tokenizer = AutoTokenizer.fromPretrained(model.toPath());
            WeightLoader wl = new DefaultWeightLoader(model);

            Constructor<? extends AbstractModel> cons = modelType.getModelClass().getConstructor(AbstractModel.InferenceType.class, Config.class,
                    WeightLoader.class, PreTrainedTokenizer.class, DType.class, DType.class, Optional.class,
                    ConfigurableTensorProvider.class, MetricRegistry.class, TensorAllocator.class,
                    KvBufferCacheSettings.class, ToolCallParser.class, WrappedForkJoinPool.class,
                    TensorParallelContext.class, TensorParallelCollectives.class);

            AbstractModel am = cons.newInstance(AbstractModel.InferenceType.FULL_CLASSIFICATION, config, wl, tokenizer,
                    workingMemoryType, workingQuantizationType, Optional.empty(), configurableTensorProvider,
                    metricRegistry, arrayQueueTensorAllocator, kvBufferCacheSettings, toolCallParser, pool,
                    new StaticTensorParallelContext(0, 1), new SingleRankTensorParallelCollectives());
            return am;
        } catch (IOException | NoSuchMethodException | InvocationTargetException | InstantiationException | IllegalAccessException e) {
            throw new RuntimeException(e);
        }
    }



    public static AbstractModel  loadEmbeddingModel(File model, DType workingMemoryType, DType workingQuantizationType,
                                                    ConfigurableTensorProvider configurableTensorProvider,
                                                    MetricRegistry metricRegistry, TensorAllocator arrayQueueTensorAllocator,
                                                    KvBufferCacheSettings kvBufferCacheSettings) {
     return load(AbstractModel.InferenceType.FULL_EMBEDDING, model, workingMemoryType, workingQuantizationType,
             configurableTensorProvider, metricRegistry, arrayQueueTensorAllocator,kvBufferCacheSettings);

    }
    protected static AbstractModel load(AbstractModel.InferenceType infType, File model, DType workingMemoryType, DType workingQuantizationType,
                                 ConfigurableTensorProvider configurableTensorProvider,
                                 MetricRegistry metricRegistry, TensorAllocator arrayQueueTensorAllocator,
                                 KvBufferCacheSettings kvBufferCacheSettings) {
        File configFile = new File(model, "config.json");
        if (!configFile.exists()){
            throw new RuntimeException("Expecting to find config file " + configFile);
        }
        ModelType modelType = detectModel(configFile);

        try {
            Config config = om.readValue(configFile, modelType.getConfigClass());
            PreTrainedTokenizer tokenizer = AutoTokenizer.fromPretrained(model.toPath());

            WeightLoader wl = new DefaultWeightLoader(model);

            Constructor<? extends AbstractModel> cons = modelType.getModelClass().getConstructor(AbstractModel.InferenceType.class, Config.class,
                    WeightLoader.class, PreTrainedTokenizer.class, DType.class, DType.class, Optional.class,
                    ConfigurableTensorProvider.class, MetricRegistry.class, TensorAllocator.class,
                    KvBufferCacheSettings.class, ToolCallParser.class, WrappedForkJoinPool.class,
                    TensorParallelContext.class, TensorParallelCollectives.class) ;

            AbstractModel am = cons.newInstance(infType, config, wl, tokenizer,
                    workingMemoryType, workingQuantizationType, Optional.empty(), configurableTensorProvider,
                    metricRegistry, arrayQueueTensorAllocator, kvBufferCacheSettings, new DefaultToolCallParser(),
                    new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores()), new StaticTensorParallelContext(0, 1),
                    new SingleRankTensorParallelCollectives());
            return am;
        } catch (IOException | NoSuchMethodException | InvocationTargetException | InstantiationException | IllegalAccessException e) {
            throw new RuntimeException(e);
        }
    }

    public static AbstractModel loadModel(
            AbstractModel.InferenceType inferenceType,
            ModelFetcher modelFetcher,
            DType workingMemoryType,
            DType workingQuantizationType,
            Optional<DType> modelQuantization,
            Optional<Integer> threadCount,
            Function<File, WeightLoader> weightLoaderSupplier) {

        File baseDir = modelFetcher.maybeDownload();
        File configFile = new File(baseDir, "config.json");
        if (!configFile.exists()){
            throw new RuntimeException("Expecting to find config file " + configFile);
        }

        try {
            //threadCount.ifPresent(PhysicalCoreExecutor::overrideThreadCount);
            ModelType modelType = detectModel(configFile);
            Config c = om.readValue(configFile, modelType.getConfigClass());
            //c.setWorkingDirectory(workingDirectory);
            PreTrainedTokenizer t = AutoTokenizer.fromPretrained(baseDir.toPath());
            WeightLoader wl = weightLoaderSupplier.apply(baseDir);

            MetricRegistry metricRegistry = new MetricRegistry();
            TensorAllocator tensorAllocator = new ArrayQueueTensorAllocator(metricRegistry);
            WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores());
            ConfigurableTensorProvider provider = new ConfigurableTensorProvider(tensorAllocator, pool);
            AbstractModel am = modelType.getModelClass()
                    .getConstructor(
                            AbstractModel.InferenceType.class,
                            Config.class,
                            WeightLoader.class,
                            PreTrainedTokenizer.class,
                            DType.class,
                            DType.class,
                            Optional.class,
                            ConfigurableTensorProvider.class,
                            MetricRegistry.class,
                            TensorAllocator.class,
                            KvBufferCacheSettings.class,
                            ToolCallParser.class,
                            WrappedForkJoinPool.class,
                            TensorParallelContext.class,
                            TensorParallelCollectives.class
                    )
                    .newInstance(inferenceType, c, wl, t, workingMemoryType, workingQuantizationType, modelQuantization,
                            provider, metricRegistry, tensorAllocator, new KvBufferCacheSettings(true), new DefaultToolCallParser(),
                            pool, new StaticTensorParallelContext(0, 1), new SingleRankTensorParallelCollectives());
            return am;

        } catch (IOException | NoSuchMethodException | InvocationTargetException | InstantiationException | IllegalAccessException e) {
            throw new RuntimeException(e);
        }
    }


    /*
    public static AbstractModel loadClassifierModel(ModelFetcher fetcher, DType workingMemoryType, DType workingQuantizationType) {
        //This callback seems so unnessary
        Function<File, WeightLoader> weightLoaderFunction = DefaultWeightLoader::new;
        return loadModel(AbstractModel.InferenceType.FULL_CLASSIFICATION,
                fetcher,
                workingMemoryType,
                workingQuantizationType,
                Optional.empty(),
                Optional.empty(),
                Optional.empty(),
                weightLoaderFunction
                );
    }*/
}
