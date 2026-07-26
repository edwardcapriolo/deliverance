package io.teknek.deliverance.safetensors;

import com.google.common.primitives.Ints;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorInfo;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.impl.BFloat16BufferTensor;
import io.teknek.deliverance.tensor.impl.Float16BufferTensor;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.operations.TensorOperations;

import java.util.Map;

/**
 * A {@link WeightLoader} decorator that merges a {@link LoraAdapter}'s low-rank deltas into the
 * delegate loader's tensors at load time.
 *
 * <p>This is the Phase 1 ("merge-at-load") LoRA support described in {@code
 * StepPlans/deliverance_lora_step3_merging_weightloader_plan_v1.md}: the merge is recomputed in
 * memory on every model load, no new files are written, and model family classes (e.g. {@code
 * LlamaModel}) are completely unaware this decorator exists -- they call {@link #load(String)}
 * exactly as they would against a {@link DefaultWeightLoader}.</p>
 */
public class MergingWeightLoader implements WeightLoader {

    private final WeightLoader delegate;
    private final LoraAdapter adapter;
    private final TensorOperations tensorOps;

    public MergingWeightLoader(WeightLoader delegate, LoraAdapter adapter, TensorOperations tensorOps) {
        this.delegate = delegate;
        this.adapter = adapter;
        this.tensorOps = tensorOps;
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
    public boolean isWeightPresent(String name) {
        return delegate.isWeightPresent(name);
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
     * TensorOperations}'s batched {@code saxpy} computes when given one row of (pre-scaled) {@code
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
     * TensorOperations#scale}: on this codebase's ARM/Panama path, {@code scale()} only implements BF16
     * for AVX_512/AVX_256 and has no native ARM kernel (unlike {@code saxpy}, which does fall back
     * correctly for BF16), so calling it here would make merging fail on BF16 base models under ARM.</p>
     */
    private AbstractTensor merge(AbstractTensor base, LoraAdapter.LoraDelta delta) {
        int outFeatures = base.shape().first();
        int inFeatures = base.shape().last();
        int rank = delta.loraA().shape().first();

        AbstractTensor merged = allocateLike(base.dType(), outFeatures, inFeatures);
        merged.copyFrom(base, 0, 0, Ints.checkedCast(base.size()));

        AbstractTensor loraA = toDType(delta.loraA(), base.dType());
        AbstractTensor scaledB = scaledCopy(delta.loraB(), base.dType(), (float) adapter.scale());

        for (int o = 0; o < outFeatures; o++) {
            AbstractTensor mergedRow = merged.slice(o);
            AbstractTensor scaledBRow = scaledB.slice(o);
            tensorOps.saxpy(scaledBRow, loraA, mergedRow, 0, 0, inFeatures, 0, 0, rank);
        }
        return merged;
    }

    private static AbstractTensor toDType(AbstractTensor src, DType target) {
        if (src.dType() == target) {
            return src;
        }
        return scaledCopy(src, target, 1.0f);
    }

    private static AbstractTensor scaledCopy(AbstractTensor src, DType target, float factor) {
        AbstractTensor converted = allocateLike(target, src.shape().first(), src.shape().last());
        for (int row = 0; row < src.shape().first(); row++) {
            for (int col = 0; col < src.shape().last(); col++) {
                converted.set(src.get(row, col) * factor, row, col);
            }
        }
        return converted;
    }

    /**
     * Duplicated from {@link DefaultWeightLoader}'s private {@code allocateLike} rather than
     * extracted into a shared utility -- see Section 11 of the step 3 plan for why this PR keeps
     * the two copies independent.
     */
    private static AbstractTensor allocateLike(DType dType, int rows, int cols) {
        TensorShape shape = TensorShape.of(rows, cols);
        return switch (dType) {
            case F32 -> new FloatBufferTensor(shape);
            case BF16 -> new BFloat16BufferTensor(shape);
            case F16 -> new Float16BufferTensor(shape);
            default -> throw new UnsupportedOperationException("Unsupported dtype for LoRA merge: " + dType);
        };
    }
}
