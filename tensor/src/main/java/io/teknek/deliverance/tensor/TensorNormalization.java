package io.teknek.deliverance.tensor;

import com.google.common.base.Preconditions;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.tensor.operations.TensorOperations;
import io.teknek.deliverance.tensorlib.TensorPlan;
import net.jafama.FastMath;

import java.util.Objects;

/** Normalization helpers composed above primitive tensor operations. */
public final class TensorNormalization {
    private TensorNormalization() {
    }

    /**
     * Applies row-wise RMSNorm to a 2D tensor using a short-lived TensorPlan.
     *
     * <p>This convenience overload preserves the requested call shape. Production code that already owns a pool should
     * prefer {@link #rmsNorm(AbstractTensor, AbstractTensor, AbstractTensor, float, TensorOperations, WrappedForkJoinPool)}
     * to avoid creating a pool per call.</p>
     */
    public static void rmsNorm(AbstractTensor output, AbstractTensor input, AbstractTensor weight, float eps,
            TensorOperations ops) {
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores())) {
            rmsNorm(output, input, weight, eps, ops, pool);
        }
    }

    /**
     * Applies row-wise RMSNorm to a 2D tensor using TensorPlan for row scheduling.
     *
     * <pre>{@code
     * invRms = 1 / sqrt(mean(input[row, :]^2) + eps)
     * output[row, col] = input[row, col] * invRms * weight[col]
     * }</pre>
     *
     * <p>If {@code weight} is {@code null}, the scale term is omitted. This is needed for DiffusionGemma's
     * self-conditioning post norm, where HF constructs `DiffusionGemmaRMSNorm(..., with_scale=False)`.</p>
     *
     * <p>The plan is intentionally composed from existing tensor primitives: row copy, elementwise multiply for squaring,
     * row sum, row scale, and optional row multiply by the norm weight. Current gap: {@link TensorOperations} does not
     * expose a provider-backed fused RMSNorm primitive, so the only scalar work left here is per-row control flow and the
     * reciprocal square-root scalar derived from the provider-backed row sum.</p>
     */
    public static void rmsNorm(AbstractTensor output, AbstractTensor input, AbstractTensor weight, float eps,
            TensorOperations ops, WrappedForkJoinPool pool) {
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(ops, "ops");
        Objects.requireNonNull(pool, "pool");
        TensorMutability.requireWritable(output, "rmsNorm");
        Preconditions.checkArgument(input.dims() == 2, "input must be 2D [rows, hidden]");
        Preconditions.checkArgument(output.dims() == 2, "output must be 2D [rows, hidden]");
        Preconditions.checkArgument(output.shape().equals(input.shape()), "output shape must match input shape");
        Preconditions.checkArgument(Float.isFinite(eps) && eps >= 0.0f, "eps must be finite and >= 0");
        int hidden = (int) input.shape().last();
        AbstractTensor normalizedWeight = null;
        AbstractTensor effectiveWeight = weight;
        try {
            if (weight != null) {
                Preconditions.checkArgument(weight.dims() == 2 && weight.shape().first() == 1
                                && weight.shape().last() >= hidden,
                        "weight must have shape [1, hidden]");
                if (weight.dType() != output.dType()) {
                    normalizedWeight = ops.quantize(weight, output.dType(), 0, hidden);
                    effectiveWeight = normalizedWeight;
                }
            }
            final AbstractTensor weightForPlan = effectiveWeight;

        try (AbstractTensor squared = output.make(TensorShape.of((int) input.shape().first(), hidden))) {
            TensorPlan plan = new TensorPlan(ops, pool).forcedRunMode(TensorPlan.RunMode.CALLER_THREAD);
            TensorPlan.FusedBuilder builder = plan.fuseRowsIntStream("rmsnorm", output.shape())
                    .read("input", plan.input("input", input))
                    .write("output", plan.mutable("output", output))
                    .read("squared", plan.mutable("squared", squared))
                    .map("output[row] = input[row]", TensorPlan.TensorOp.CUSTOM, (ctx, rowOffset, rowLength) -> {
                        AbstractTensor in = ctx.tensor("input");
                        AbstractTensor out = ctx.tensor("output");
                        int row = (int) rowOffset;
                        out.copyFrom(in, in.getOffset(row, 0), out.getOffset(row, 0), hidden);
                    })
                    .map("squared[row] = input[row]", TensorPlan.TensorOp.CUSTOM, (ctx, rowOffset, rowLength) -> {
                        AbstractTensor in = ctx.tensor("input");
                        AbstractTensor square = ctx.tensor("squared");
                        int row = (int) rowOffset;
                        square.copyFrom(in, in.getOffset(row, 0), square.getOffset(row, 0), hidden);
                    })
                    .map("squared[row] *= input[row]", TensorPlan.TensorOp.MUL_IN_PLACE,
                            (ctx, rowOffset, rowLength) -> {
                        AbstractTensor in = ctx.tensor("input");
                        AbstractTensor square = ctx.tensor("squared");
                        try (AbstractTensor inRow = in.slice((int) rowOffset);
                             AbstractTensor squareRow = square.slice((int) rowOffset)) {
                            ops.maccumulate(squareRow, inRow, 0, hidden);
                        }
                    })
                    .map("output[row] *= inv_rms(sum(squared[row]))", TensorPlan.TensorOp.CUSTOM,
                            (ctx, rowOffset, rowLength) -> {
                        AbstractTensor out = ctx.tensor("output");
                        AbstractTensor square = ctx.tensor("squared");
                        int row = (int) rowOffset;
                        float sumSquares = ops.sum(square, row, 0, hidden);
                        float invRms = (float) (1.0 / FastMath.sqrt(sumSquares / hidden + eps));
                        try (AbstractTensor outRow = out.slice(row)) {
                            ops.scale(invRms, outRow, 0, hidden);
                        }
                    });
            if (weightForPlan != null) {
                builder.map("output[row] *= weight", TensorPlan.TensorOp.MUL_IN_PLACE,
                        (ctx, rowOffset, rowLength) -> {
                    AbstractTensor out = ctx.tensor("output");
                    try (AbstractTensor outRow = out.slice((int) rowOffset)) {
                        ops.maccumulate(outRow, weightForPlan, 0, hidden);
                    }
                });
            }
            builder.tensor().materialize();
        }
        } finally {
            if (normalizedWeight != null) {
                normalizedWeight.close();
            }
        }
    }

    /**
     * Applies RMSNorm across the last dimension of a 3D tensor.
     *
     * <p>The input and output are shaped {@code [batch, rows, hidden]}. This method flattens the leading dimensions into
     * a temporary 2D tensor {@code [batch * rows, hidden]}, delegates to the TensorPlan-backed 2D
     * {@link #rmsNorm(AbstractTensor, AbstractTensor, AbstractTensor, float, TensorOperations, WrappedForkJoinPool)}, then
     * copies the normalized rows back into the 3D output. This keeps the RMSNorm formula in one place while supporting
     * DiffusionGemma canvas tensors.</p>
     */
    public static void rmsNormLastDim(AbstractTensor output, AbstractTensor input, AbstractTensor weight, float eps,
            TensorOperations ops, WrappedForkJoinPool pool) {
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(input, "input");
        TensorMutability.requireWritable(output, "rmsNormLastDim");
        Preconditions.checkArgument(input.dims() == 3, "input must be 3D [batch, rows, hidden]");
        Preconditions.checkArgument(output.dims() == 3, "output must be 3D [batch, rows, hidden]");
        Preconditions.checkArgument(output.shape().equals(input.shape()), "output shape must match input shape");
        int batchSize = (int) input.shape().dim(0);
        int rows = (int) input.shape().dim(1);
        int hidden = (int) input.shape().dim(2);
        int flattenedRows = batchSize * rows;
        try (AbstractTensor flatInput = input.make(TensorShape.of(flattenedRows, hidden));
             AbstractTensor flatOutput = output.make(TensorShape.of(flattenedRows, hidden))) {
            TensorPlan flattenPlan = new TensorPlan(ops, pool).forcedRunMode(TensorPlan.RunMode.CALLER_THREAD);
            flattenPlan.fuseRowsIntStream("rmsnorm.flatten", flatInput.shape())
                    .read("input", flattenPlan.input("input", input))
                    .write("flatInput", flattenPlan.mutable("flatInput", flatInput))
                    .map("flatInput[row] = input[batch, position]", TensorPlan.TensorOp.CUSTOM,
                            (ctx, rowOffset, rowLength) -> {
                        int flatRow = (int) rowOffset;
                        int batch = flatRow / rows;
                        int position = flatRow % rows;
                        AbstractTensor in = ctx.tensor("input");
                        AbstractTensor flat = ctx.tensor("flatInput");
                        flat.copyFrom(in, in.getOffset(batch, position, 0), flat.getOffset(flatRow, 0), hidden);
                    })
                    .tensor()
                    .materialize();
            AbstractTensor normalizedFlatInput = null;
            try {
                normalizedFlatInput = flatInput.dType() == flatOutput.dType()
                        ? null
                        : ops.quantize(flatInput, flatOutput.dType(), 0, hidden);
                rmsNorm(flatOutput, normalizedFlatInput == null ? flatInput : normalizedFlatInput, weight, eps, ops,
                        pool);
            } finally {
                if (normalizedFlatInput != null) {
                    normalizedFlatInput.close();
                }
            }
            TensorPlan inflatePlan = new TensorPlan(ops, pool).forcedRunMode(TensorPlan.RunMode.CALLER_THREAD);
            inflatePlan.fuseRowsIntStream("rmsnorm.inflate", flatOutput.shape())
                    .read("flatOutput", inflatePlan.input("flatOutput", flatOutput))
                    .write("output", inflatePlan.mutable("output", output))
                    .map("output[batch, position] = flatOutput[row]", TensorPlan.TensorOp.CUSTOM,
                            (ctx, rowOffset, rowLength) -> {
                        int flatRow = (int) rowOffset;
                        int batch = flatRow / rows;
                        int position = flatRow % rows;
                        AbstractTensor flat = ctx.tensor("flatOutput");
                        AbstractTensor out = ctx.tensor("output");
                        out.copyFrom(flat, flat.getOffset(flatRow, 0), out.getOffset(batch, position, 0), hidden);
                    })
                    .tensor()
                    .materialize();
        }
    }
}
