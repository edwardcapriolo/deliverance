package io.teknek.reckoning.safetensors;

import com.google.common.primitives.Ints;
import io.teknek.reckoning.DType;
import io.teknek.reckoning.tensor.AbstractTensor;
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
    }

    @Override
    public TensorRef loadRef(String name) {
        Optional<LoraAdapter.LoraDelta> delta = adapter.deltaFor(name);
        TensorRef base = delegate.loadRef(name);
        if (delta.isEmpty()) {
            return base;
        }
        try {
            return mergeRef(base, delta.get());
        } finally {
            base.close();
        }
    }

    @Override
    public TensorRef loadRef(String name, TensorShardSpec shardSpec) {
        if (adapter.deltaFor(name).isPresent()) {
            throw new UnsupportedOperationException(
                    "TensorRef loading does not support merged LoRA weights: " + name);
        }
        return delegate.loadRef(name, shardSpec);
    }

    @Override
    public AbstractTensor load(String name) {
        AbstractTensor base = delegate.load(name);
        return adapter.deltaFor(name).map(delta -> merge(base, delta)).orElse(base);
    }

    @Override
    public AbstractTensor loadRows(String name, int rowOffset, int rowCount) {
        if (adapter.deltaFor(name).isPresent()) {
            throw new UnsupportedOperationException(
                    "LoRA merge-at-load does not support partial-row loading for targeted module: " + name);
        }
        return delegate.loadRows(name, rowOffset, rowCount);
    }

    @Override
    public AbstractTensor load(String name, TensorShardSpec shardSpec) {
        if (adapter.deltaFor(name).isPresent()) {
            throw new UnsupportedOperationException(
                    "LoRA merge-at-load does not support tensor-parallel sharded loading for targeted module: "
                            + name + " -- tensor-parallel + LoRA is an explicit non-goal for this PR");
        }
        return delegate.load(name, shardSpec);
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
     * Computes {@code base + scale * (loraB @ loraA)} and returns a freshly allocated tensor;
     * {@code base} is left untouched.
     *
     * <p>{@code base} has shape {@code [out, in]}. {@code loraA} has shape {@code [rank, in]} and
     * {@code loraB} has shape {@code [out, rank]} (standard PEFT layout). For each output row
     * {@code o}, {@code delta[o,:] = scale * sum_k loraB[o,k] * loraA[k,:]} -- exactly what {@link
     * TensorRef}'s batched {@code saxpy} computes when given one row of (pre-scaled) {@code
     * loraB} as {@code alpha} and all of {@code loraA} as {@code x}. The batched primitive requires
     * both its {@code alpha} and {@code y} arguments to be single-row tensors, so this loops over
     * output rows, slicing one row of {@code merged} (as {@code y}) and one row of {@code scaledB}
     * (as {@code alpha}) per iteration -- both slices are views over the allocated tensors, not
     * copies, so the accumulation writes straight into {@code merged}.</p>
     *
     * <p>{@code loraA}/{@code loraB} are converted to {@code base}'s dtype first: the underlying
     * {@code saxpy} implementations only support matching dtypes, or a BF16 {@code x} against an F32
     * {@code y}, and a real-world adapter's stored dtype (commonly F32) has no guaranteed relationship
     * to the base model's dtype (commonly BF16 for HF-distributed dense checkpoints). {@code loraB} is
     * pre-scaled by {@code adapter.scale()} during that same conversion pass rather than via {@link
     * {@code scale}: on this codebase's ARM/Panama path, {@code scale()} only implements BF16
     * for AVX_512/AVX_256 and has no native ARM kernel (unlike {@code saxpy}, which does fall back
     * correctly for BF16), so calling it here would make merging fail on BF16 base models under ARM.</p>
     */
    private AbstractTensor merge(AbstractTensor base, LoraAdapter.LoraDelta delta) {
        int outFeatures = base.shape().first();
        int inFeatures = base.shape().last();
        int rank = delta.loraA().shape().first();

        AbstractTensor merged = LoraTensorMath.allocateLike(base.dType(), outFeatures, inFeatures);
        merged.copyFrom(base, 0, 0, Ints.checkedCast(base.size()));

        AbstractTensor loraA = LoraTensorMath.toDType(delta.loraA(), base.dType());
        AbstractTensor scaledB = LoraTensorMath.scaledCopy(delta.loraB(), base.dType(), (float) adapter.scale());

        for (int o = 0; o < outFeatures; o++) {
            AbstractTensor mergedRow = merged.slice(o);
            AbstractTensor scaledBRow = scaledB.slice(o);
            try (TensorRef scaledBRef = TensorRef.borrowed(scaledBRow);
                 TensorRef loraARef = TensorRef.borrowed(loraA);
                 TensorRef mergedRef = TensorRef.borrowed(mergedRow)) {
                lighter.saxpy(scaledBRef, loraARef, mergedRef, 0, 0, inFeatures, 0, 0, rank);
            }
        }
        return merged;
    }

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
        AbstractTensor loraA = LoraTensorMath.toDType(delta.loraA(), base.dType());
        AbstractTensor scaledB = LoraTensorMath.scaledCopy(delta.loraB(), base.dType(), (float) adapter.scale());
        boolean closeLoraA = loraA != delta.loraA();
        try {
            lighter.copy(base, 0, merged, 0, Math.toIntExact(base.shape().size()));
            try (TensorRef loraARef = TensorRef.borrowed(loraA);
                 TensorRef scaledBRef = TensorRef.borrowed(scaledB)) {
                if (base.dType() == DType.F32 || base.dType() == DType.BF16) {
                    try (TensorRef transposedA = lighter.allocate(base.dType(), TensorShape.of(inFeatures, rank))) {
                        lighter.transpose(new Transpose(loraARef).destination(transposedA));
                        compositeOps.gemmAccumulate(new GemmAccumulate(merged, scaledBRef, transposedA, 1.0f));
                    }
                } else {
                    for (int row = 0; row < outFeatures; row++) {
                        try (TensorRef mergedRow = merged.slice(row);
                             TensorRef scaledBRow = scaledBRef.slice(row)) {
                            lighter.saxpy(scaledBRow, loraARef, mergedRow, 0, 0, inFeatures, 0, 0, rank);
                        }
                    }
                }
            }
            return merged;
        } catch (RuntimeException | Error e) {
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
