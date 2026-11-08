package io.teknek.reckoning.safetensors;

import io.teknek.reckoning.DType;
import io.teknek.reckoning.tensor2.Lighter;
import io.teknek.reckoning.tensor2.TensorRef;

import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Wraps a {@link LoraAdapter} for Phase 2 ("runtime hot-swap") use, lazily resolving and caching
 * one {@link LoraLayerDelta} per base tensor name.
 *
 * <p>{@link LoraAdapter#deltaFor(String)} does real work on every call (a module-suffix parse, a
 * set lookup, and -- if targeted -- two fresh {@code weights.load(...)} calls). Phase 1
 * (merge-at-load) only ever pays this once per tensor, at load time; this class gives Phase 2 the
 * same "pay once" property across every later token/request for a registered adapter, by caching
 * the resolved, dtype-converted, pre-scaled result (including the {@code Optional.empty()} case
 * for non-targeted names) behind a {@link ConcurrentHashMap}. See step 4 plan Section 3.2.</p>
 */
public class ResolvedLoraAdapter implements AutoCloseable {

    private final LoraAdapter adapter;
    private final DType targetDType;
    private final Lighter lighter;
    private final ConcurrentHashMap<String, Optional<LoraLayerDelta>> cache = new ConcurrentHashMap<>();

    public ResolvedLoraAdapter(LoraAdapter adapter, DType targetDType, Lighter lighter) {
        this.adapter = adapter;
        this.targetDType = targetDType;
        this.lighter = java.util.Objects.requireNonNull(lighter, "lighter");
        adapter.setLighter(lighter);
    }

    public Optional<LoraLayerDelta> deltaFor(String baseTensorName) {
        return cache.computeIfAbsent(baseTensorName, name -> adapter.deltaFor(name).map(this::resolve));
    }

    private LoraLayerDelta resolve(LoraAdapter.LoraDelta delta) {
        TensorRef sourceA = delta.loraA();
        TensorRef sourceB = delta.loraB();
        TensorRef loraA = sourceA;
        int rank = Math.toIntExact(sourceA.shape().first());
        try {
            loraA = LoraTensorMath.toDType(sourceA, targetDType, lighter);
            TensorRef scaledLoraB = LoraTensorMath.scaledCopy(sourceB, targetDType,
                    (float) adapter.scale(), lighter);
            if (loraA != sourceA) {
                sourceA.close();
            }
            sourceB.close();
            return new LoraLayerDelta(loraA, scaledLoraB, rank);
        } catch (RuntimeException e) {
            if (loraA != sourceA) {
                loraA.close();
            } else {
                sourceA.close();
            }
            sourceB.close();
            throw e;
        }
    }

    @Override
    public void close() {
        cache.values().forEach(value -> value.ifPresent(LoraLayerDelta::close));
        cache.clear();
        adapter.close();
    }
}
