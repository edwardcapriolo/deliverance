package io.teknek.reckoning.safetensors;

import io.teknek.reckoning.DType;
import io.teknek.reckoning.tensor.TensorInfo;
import io.teknek.reckoning.tensor.TensorShape;
import io.teknek.reckoning.tensor2.CompositeOps;
import io.teknek.reckoning.tensor2.GemmAccumulate;
import io.teknek.reckoning.tensor2.Lighter;
import io.teknek.reckoning.tensor2.Transpose;
import io.teknek.reckoning.tensor2.TensorRef;

import java.util.Map;
import java.nio.file.Path;
import java.util.Objects;
import java.util.Optional;

/**
 * A {@link WeightLoader} decorator that merges a {@link LoraAdapter}'s low-rank deltas into the
 * delegate loader's tensors at load time.
 *
 * <p>This is the Phase 1 ("merge-at-load") LoRA support described in {@code
 * StepPlans/reckoning_lora_step3_merging_weightloader_plan_v1.md}: the merge is recomputed in
 * memory on every model load, no new files are written, and model family classes (e.g. {@code
 * LlamaModel}) are completely unaware this decorator exists -- they call {@link #load(String)}
 * exactly as they would against a {@link DefaultWeightLoader}.</p>
 */
public class MergingWeightLoader implements WeightLoader {

    private final WeightLoader delegate;
    private final LoraAdapter adapter;
    private final Lighter lighter;
    private final CompositeOps compositeOps;

    public MergingWeightLoader(WeightLoader delegate, LoraAdapter adapter, Lighter lighter) {
        this.delegate = delegate;
        this.adapter = adapter;
        this.lighter = Objects.requireNonNull(lighter, "lighter");
        this.compositeOps = new CompositeOps(lighter);
        delegate.setLighter(lighter);
        adapter.setLighter(lighter);
        DType baseDType = delegate.getModelDType();
        if (baseDType != DType.F32 && baseDType != DType.BF16 && baseDType != DType.F16) {
            throw new UnsupportedOperationException(
                    "LoRA merge-at-load requires a dense (F32/BF16/F16) base model; got " + baseDType
                            + " -- pre-quantized-on-disk base models are not supported for merge-at-load");
        }
    }

    @Override
    public Map<String, String> metadata() {
        return delegate.metadata();
    }

    @Override
    public Map<String, TensorInfo> tensorInfoMap() {
        return delegate.tensorInfoMap();
    }

    @Override
    public DType getModelDType() {
        return delegate.getModelDType();
    }

    @Override
    public void setUseMajorityDType(boolean useMajorityDType) {
        delegate.setUseMajorityDType(useMajorityDType);
    }

    @Override
    public Optional<Path> modelRoot() {
        return delegate.modelRoot();
    }

    @Override
    public boolean isWeightPresent(String name) {
        return delegate.isWeightPresent(name);
    }

    @Override
    public void setLighter(Lighter lighter) {
        delegate.setLighter(lighter);
        adapter.setLighter(lighter);
    }

    @Override
    public TensorRef load(String name) {
        Optional<LoraAdapter.LoraDelta> delta = adapter.deltaFor(name);
        TensorRef base = delegate.load(name);
        if (delta.isEmpty()) {
            return base;
        }
        try (LoraAdapter.LoraDelta resolved = delta.get()) {
            try {
                return mergeRef(base, resolved);
            } finally {
                base.close();
            }
        }
    }

    @Override
    public TensorRef load(String name, TensorShardSpec shardSpec) {
        Optional<LoraAdapter.LoraDelta> delta = adapter.deltaFor(name);
        if (delta.isPresent()) {
            delta.get().close();
            throw new UnsupportedOperationException(
                    "TensorRef loading does not support merged LoRA weights: " + name);
        }
        return delegate.load(name, shardSpec);
    }

    @Override
    public TensorRef loadRows(String name, int rowOffset, int rowCount) {
        Optional<LoraAdapter.LoraDelta> delta = adapter.deltaFor(name);
        if (delta.isPresent()) {
            delta.get().close();
            throw new UnsupportedOperationException(
                    "LoRA merge-at-load does not support partial-row loading for targeted module: " + name);
        }
        return delegate.loadRows(name, rowOffset, rowCount);
    }

    @Override
    public TensorRef loadRef(String name) {
        return load(name);
    }

    @Override
    public TensorRef loadRef(String name, TensorShardSpec shardSpec) {
        return load(name, shardSpec);
    }

    /** Closes both the delegate loader and the adapter -- this loader is the adapter's sole owner. */
    @Override
    public void close() throws Exception {
        try {
            delegate.close();
        } finally {
            adapter.close();
        }
    }

    /**
     * Computes {@code base + scale * (loraB @ loraA)} into a freshly allocated TensorRef;
     * {@code base} is left untouched.
     *
     * <p>{@code base} has shape {@code [out, in]}. {@code loraA} has shape {@code [rank, in]} and
     * {@code loraB} has shape {@code [out, rank]} (standard PEFT layout). The adapter scale is
     * applied to {@code loraB} before accumulation. Dense F32/BF16 bases use the fused GEMM path;
     * F16 and other supported dense paths use row-wise TensorRef saxpy.</p>
     *
     * <p>LoRA weights are converted to the base dtype before the provider operation. This is
     * required because the saxpy implementations require compatible dtypes, while adapter weights
     * are commonly stored as F32 and the base model may be BF16. Scaling is performed through the
     * TensorRef Lighter scale operation rather than a separate scalar conversion path so the
     * Panama and ARM providers share the same behavior.</p>
     */
    private TensorRef mergeRef(TensorRef base, LoraAdapter.LoraDelta delta) {
        int outFeatures = Math.toIntExact(base.shape().first());
        int inFeatures = Math.toIntExact(base.shape().last());
        int rank = Math.toIntExact(delta.loraA().shape().first());
        if (delta.loraA().shape().last() != inFeatures
                || delta.loraB().shape().first() != outFeatures
                || delta.loraB().shape().last() != rank) {
            throw new IllegalArgumentException("LoRA shape does not match base weight: base=" + base.shape()
                    + " loraA=" + delta.loraA().shape() + " loraB=" + delta.loraB().shape());
        }

        TensorRef merged = lighter.allocate(base.dType(), base.shape());
        TensorRef loraA = LoraTensorMath.toDType(delta.loraA(), base.dType(), lighter);
        TensorRef scaledB = LoraTensorMath.scaledCopy(delta.loraB(), base.dType(),
                (float) adapter.scale(), lighter);
        boolean closeLoraA = loraA != delta.loraA();
        try {
            lighter.copy(base, 0, merged, 0, Math.toIntExact(base.shape().size()));
            if (base.dType() == DType.F32 || base.dType() == DType.BF16) {
                try (TensorRef transposedA = lighter.allocate(base.dType(), TensorShape.of(inFeatures, rank))) {
                    lighter.transpose(new Transpose(loraA).destination(transposedA));
                    compositeOps.gemmAccumulate(new GemmAccumulate(merged, scaledB, transposedA, 1.0f));
                }
            } else {
                for (int row = 0; row < outFeatures; row++) {
                    try (TensorRef mergedRow = merged.slice(row);
                         TensorRef scaledBRow = scaledB.slice(row)) {
                        lighter.saxpy(scaledBRow, loraA, mergedRow, 0, 0, inFeatures, 0, 0, rank);
                    }
                }
            }
            return merged;
        } catch (RuntimeException e) {
            merged.close();
            throw e;
        } finally {
            scaledB.close();
            if (closeLoraA) {
                loraA.close();
            }
        }
    }
}
