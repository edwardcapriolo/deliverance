package io.teknek.deliverance.safetensors;

import com.codahale.metrics.MetricRegistry;
import com.codahale.metrics.Timer;
import io.teknek.deliverance.safetensors.fetch.LoraAdapterModelFetcher;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorInfo;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashSet;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/**
 * A parsed, in-memory HuggingFace PEFT-format LoRA adapter (a directory containing {@code
 * adapter_config.json} and {@code adapter_model.safetensors}).
 *
 * <p>This is intentionally a self-contained data holder with no ties to model loading: it
 * knows how to read an adapter and answer "what is the low-rank delta for base tensor X", but
 * nothing in {@code core} references it yet. Merging these deltas into a base model's weights
 * (Phase 1) or applying them at inference time (Phase 2) are separate, later pieces of work.</p>
 *
 * <p>Adapters are small (typically single-digit-MB to a few tens of MB for common ranks), so
 * unlike {@link DefaultWeightLoader} this does not need multi-file/mmap-split handling — the
 * whole adapter file is mapped and wrapped in a single {@link Weights} instance.</p>
 */
public class LoraAdapter implements AutoCloseable {

    public static final String SAFETENSORS_FILE_NAME = "adapter_model.safetensors";

    private final LoraAdapterConfig config;
    private final Set<String> targetModuleSet;
    private final Weights weights;
    private final RandomAccessFile file;
    private final MetricRegistry metricRegistry;

    private LoraAdapter(LoraAdapterConfig config, Weights weights, RandomAccessFile file, MetricRegistry metricRegistry) {
        this.config = config;
        this.targetModuleSet = new HashSet<>(config.targetModules);
        this.weights = weights;
        this.file = file;
        this.metricRegistry = metricRegistry;
        validateTargetModulesPresent();
    }

    public static LoraAdapter load(File adapterDir) {
        return load(adapterDir, new MetricRegistry());
    }

    public static LoraAdapter load(File adapterDir, MetricRegistry metricRegistry) {
        LoraAdapterConfig config = LoraAdapterConfig.load(adapterDir);
        File safetensorsFile = new File(adapterDir, SAFETENSORS_FILE_NAME);

        try (Timer.Context ignored = metricRegistry.timer("loraadapter.parse_header").time()) {
            RandomAccessFile raf = new RandomAccessFile(safetensorsFile, "r");
            try {
                ByteBuffer mapped = raf.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, raf.length());
                Map<String, String> metadata = new java.util.HashMap<>();
                Map<String, TensorInfo> tensorInfoMap = DefaultWeightLoader.readTensorInfoMap(mapped, Optional.of(metadata));
                Weights weights = new Weights(metadata, tensorInfoMap, mapped, Optional.empty());
                return new LoraAdapter(config, weights, raf, metricRegistry);
            } catch (RuntimeException | Error e) {
                closeQuietly(raf);
                throw e;
            }
        } catch (IOException e) {
            throw new UncheckedIOException("Unable to read " + safetensorsFile, e);
        }
    }

    public static LoraAdapter fromPretrained(LoraAdapterModelFetcher fetcher) {
        return fromPretrained(fetcher, new MetricRegistry());
    }

    public static LoraAdapter fromPretrained(LoraAdapterModelFetcher fetcher, MetricRegistry metricRegistry) {
        File adapterDir;
        try (Timer.Context ignored = metricRegistry.timer("loraadapter.fetch").time()) {
            adapterDir = fetcher.maybeDownload();
        }
        return load(adapterDir, metricRegistry);
    }

    public int rank() {
        return config.rank;
    }

    public double alpha() {
        return config.alpha;
    }

    /** The standard LoRA merge scale, {@code alpha / r}. */
    public double scale() {
        return config.scale();
    }

    /**
     * Returns the low-rank delta for the given base tensor, or empty if this adapter does not
     * target that tensor's module.
     *
     * <p>Note: presence of the module type in {@code target_modules} is validated eagerly at
     * construction (see {@link #validateTargetModulesPresent()}), but this PR does not model
     * PEFT's {@code layers_to_transform}/{@code rank_pattern} fields, so a specific layer
     * within an otherwise-targeted module could still be absent from the adapter file. In that
     * case this throws {@link java.util.NoSuchElementException} from the underlying {@link
     * Weights#load}, rather than silently returning empty — a missing tensor for a supposedly
     * targeted module indicates a real mismatch worth surfacing, not a normal "not targeted"
     * case.</p>
     */
    public Optional<LoraDelta> deltaFor(String baseTensorName) {
        Optional<String> module = LoraTensorNames.moduleSuffix(baseTensorName);
        if (module.isEmpty() || !targetModuleSet.contains(module.get())) {
            return Optional.empty();
        }
        try (Timer.Context ignored = metricRegistry.timer("loraadapter.load_tensor").time()) {
            AbstractTensor loraA = weights.load(LoraTensorNames.loraA(baseTensorName));
            AbstractTensor loraB = weights.load(LoraTensorNames.loraB(baseTensorName));
            return Optional.of(new LoraDelta(loraA, loraB));
        }
    }

    /**
     * Sanity-checks, at construction time, that every {@code target_modules} entry actually
     * appears in the adapter file (as at least one {@code lora_A}/{@code lora_B} pair), and
     * that the first such pair found for each module agrees with the config's rank. This
     * catches a truncated download or a mismatched config immediately, rather than failing
     * deep inside a later merge/forward call with a generic "tensor not found" error.
     *
     * <p>This only samples one occurrence per module (typically layer 0) — full per-layer
     * cross-checking against an actual base model's dimensions is out of scope for this PR
     * (see the parent plan) and belongs to Phase 1's {@code MergingWeightLoader}, which has
     * both the adapter and the base model to compare.</p>
     */
    private void validateTargetModulesPresent() {
        Map<String, TensorInfo> tensorInfoMap = weights.tensorInfoMap();
        Set<String> modulesFound = new HashSet<>();
        for (String tensorName : tensorInfoMap.keySet()) {
            adapterModuleName(tensorName).ifPresent(modulesFound::add);
        }
        for (String module : config.targetModules) {
            if (!modulesFound.contains(module)) {
                throw new IllegalStateException(
                        "LoRA adapter config targets module \"" + module + "\" but no matching lora_A/lora_B "
                                + "tensors were found in " + SAFETENSORS_FILE_NAME);
            }
        }
        for (String module : config.targetModules) {
            validateSampleShape(module, tensorInfoMap);
        }
    }

    private void validateSampleShape(String module, Map<String, TensorInfo> tensorInfoMap) {
        for (String tensorName : tensorInfoMap.keySet()) {
            if (!tensorName.endsWith(".lora_A.weight")) {
                continue;
            }
            if (!module.equals(adapterModuleName(tensorName).orElse(null))) {
                continue;
            }
            TensorInfo aInfo = tensorInfoMap.get(tensorName);
            String bTensorName = tensorName.substring(0, tensorName.length() - ".lora_A.weight".length()) + ".lora_B.weight";
            TensorInfo bInfo = tensorInfoMap.get(bTensorName);
            if (bInfo == null) {
                throw new IllegalStateException("LoRA adapter has " + tensorName + " but no matching " + bTensorName);
            }
            int aRank = aInfo.shape[0];
            int bRank = bInfo.shape[bInfo.shape.length - 1];
            if (aRank != config.rank || bRank != config.rank) {
                throw new IllegalStateException(
                        "LoRA adapter config declares rank " + config.rank + " but " + tensorName + "/" + bTensorName
                                + " have ranks " + aRank + "/" + bRank);
            }
            return;
        }
    }

    private static Optional<String> adapterModuleName(String adapterTensorName) {
        String suffix = null;
        if (adapterTensorName.endsWith(".lora_A.weight")) {
            suffix = ".lora_A.weight";
        } else if (adapterTensorName.endsWith(".lora_B.weight")) {
            suffix = ".lora_B.weight";
        } else {
            return Optional.empty();
        }
        String withoutSuffix = adapterTensorName.substring(0, adapterTensorName.length() - suffix.length());
        int lastDot = withoutSuffix.lastIndexOf('.');
        return Optional.of(lastDot < 0 ? withoutSuffix : withoutSuffix.substring(lastDot + 1));
    }

    private static void closeQuietly(RandomAccessFile raf) {
        try {
            raf.close();
        } catch (IOException ignored) {
            // best-effort cleanup on the construction-failure path
        }
    }

    @Override
    public void close() {
        try {
            file.close();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    @Override
    public String toString() {
        return "LoraAdapter{" + config + '}';
    }

    public record LoraDelta(AbstractTensor loraA, AbstractTensor loraB) {
    }
}
