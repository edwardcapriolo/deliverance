package net.deliverance.http;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.JsonUtils;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelConfig;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.model.CausalLanguageModel;
import io.teknek.deliverance.model.ModelSupport;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.dropwizard.metrics5.MetricRegistry;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;

import java.io.File;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.Semaphore;

@RestController
public class ServeController {

    private final Map<MultiModelConfig, AbstractModel> embeddingModels;
    private final Map<MultiModelConfig, CausalLanguageModel> causalLanguageModels;
    private final ConfigurableTensorProvider provider;
    private final MetricRegistry metricRegistry;
    private final TensorAllocator tensorAllocator;
    private final WrappedForkJoinPool pool;
    private final Semaphore serveGate = new Semaphore(1);

    public ServeController(
            @Qualifier("embeddingModels") Map<MultiModelConfig, AbstractModel> embeddingModels,
            @Qualifier("causalLanguageModels") Map<MultiModelConfig, CausalLanguageModel> causalLanguageModels,
            ConfigurableTensorProvider provider,
            MetricRegistry metricRegistry,
            TensorAllocator tensorAllocator,
            WrappedForkJoinPool pool) {
        this.embeddingModels = embeddingModels;
        this.causalLanguageModels = causalLanguageModels;
        this.provider = provider;
        this.metricRegistry = metricRegistry;
        this.tensorAllocator = tensorAllocator;
        this.pool = pool;
    }

    @RequestMapping(method = RequestMethod.POST, value = "/api/serve", consumes = "application/json", produces = "application/json")
    public ResponseEntity<ServeResponse> serve(@RequestBody ServeRequest request) {
        if (request == null || isBlank(request.owner()) || isBlank(request.name()) || request.type() == null) {
            return ResponseEntity.badRequest().body(ServeResponse.failed("owner, name, and type are required"));
        }
        if (!serveGate.tryAcquire()) {
            return ResponseEntity.status(HttpStatus.TOO_MANY_REQUESTS)
                    .header(HttpHeaders.RETRY_AFTER, "5")
                    .body(ServeResponse.failed("another model is being served"));
        }
        try {
            MultiModelConfig config = configFor(request);
            if (modelMap(request.type()).containsKey(config)) {
                return ResponseEntity.ok(ServeResponse.ready(request.owner(), request.name(), request.type()));
            }

            ModelFetcher fetcher = new ModelFetcher(request.owner(), request.name()).withDownload(false);
            if (request.type() == ServeType.EMBEDDING) {
                File modelDirectory = fetcher.maybeDownload();
                AbstractModel model = ModelSupport.loadEmbeddingModel(
                        modelDirectory,
                        DType.F32,
                        DType.I8,
                        provider,
                        metricRegistry,
                        tensorAllocator,
                        new KvBufferCacheSettings(true));
                embeddingModels.put(config, model);
            } else {
                AutoModelForCausaLm.Builder builder = AutoModelForCausaLm.newBuilder(fetcher)
                        .withMetricRegistry(metricRegistry)
                        .withTensorAllocator(tensorAllocator)
                        .withPrimaryTensorProvider(provider)
                        .withWrappedForkJoinPool(pool);
                if (request.config() != null && !request.config().isEmpty()) {
                    builder.withConfig(JsonUtils.om.convertValue(request.config(), AutoModelConfig.class));
                }
                builder.withDownload(false);
                causalLanguageModels.put(config, builder.build());
            }
            return ResponseEntity.ok(ServeResponse.ready(request.owner(), request.name(), request.type()));
        } catch (IllegalStateException e) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND)
                    .body(ServeResponse.failed(e.getMessage()));
        } finally {
            serveGate.release();
        }
    }

    private static MultiModelConfig configFor(ServeRequest request) {
        MultiModelConfig config = new MultiModelConfig();
        config.setModelOwner(request.owner());
        config.setModelName(request.name());
        config.setInferenceType(request.type().name());
        return config;
    }

    private Map<?, ?> modelMap(ServeType type) {
        return type == ServeType.EMBEDDING ? embeddingModels : causalLanguageModels;
    }

    private static boolean isBlank(String value) {
        return value == null || value.isBlank();
    }

    public enum ServeType {
        EMBEDDING,
        GENERATION
    }

    public record ServeRequest(String owner, String name, ServeType type, Map<String, Object> config) {
    }

    public enum ServeStatus {
        READY,
        FAILED
    }

    public record ServeResponse(ServeStatus status, String owner, String name, ServeType type, String error) {
        static ServeResponse ready(String owner, String name, ServeType type) {
            return new ServeResponse(ServeStatus.READY, owner, name, type, null);
        }

        static ServeResponse failed(String error) {
            return new ServeResponse(ServeStatus.FAILED, null, null, null, Objects.requireNonNull(error));
        }
    }
}
