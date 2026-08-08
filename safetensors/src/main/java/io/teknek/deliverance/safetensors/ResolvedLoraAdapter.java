package io.teknek.deliverance.safetensors;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.AbstractTensor;

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
    private final ConcurrentHashMap<String, Optional<LoraLayerDelta>> cache = new ConcurrentHashMap<>();

    public ResolvedLoraAdapter(LoraAdapter adapter, DType targetDType) {
        this.adapter = adapter;
        this.targetDType = targetDType;
    }

    public Optional<LoraLayerDelta> deltaFor(String baseTensorName) {
        return cache.computeIfAbsent(baseTensorName, name -> adapter.deltaFor(name).map(this::resolve));
    }

    private LoraLayerDelta resolve(LoraAdapter.LoraDelta delta) {
        AbstractTensor loraA = LoraTensorMath.toDType(delta.loraA(), targetDType);
        AbstractTensor scaledLoraB = LoraTensorMath.scaledCopy(delta.loraB(), targetDType, (float) adapter.scale());
        return new LoraLayerDelta(loraA, scaledLoraB, delta.loraA().shape().first());
    }

    @Override
    public void close() {
        adapter.close();
    }
}
