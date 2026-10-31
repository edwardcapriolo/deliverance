package io.teknek.deliverance.model.diffusiongemma;

import com.google.common.base.Preconditions;
import io.dropwizard.metrics5.MetricRegistry;
import io.dropwizard.metrics5.Timer;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.TensorNormalization;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.operations.TensorOperations;
import io.teknek.deliverance.tensorlib.TensorPlan;
import io.teknek.deliverance.tensor2.Accumulate;
import io.teknek.deliverance.tensor2.ActivationMultiplyQuantize;
import io.teknek.deliverance.tensor2.CompositeOps;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;

import java.util.Objects;

/**
 * DiffusionGemma self-conditioning block, composed from TensorPlan/tensor primitives.
 *
 * <p>Matches the HF data flow:</p>
 *
 * <pre>{@code
 * normed = pre_norm(self_conditioning_signal)
 * sc_signal = down_proj(act(gate_proj(normed)) * up_proj(normed))
 * combined = inputs_embeds + sc_signal
 * output = post_norm(combined)
 * }</pre>
 *
 * <p>Current gap: flattening between 3D canvas tensors and 2D projection tensors still copies data. TensorPlan can later
 * fuse these layout transitions; this class keeps the math expressed through existing tensor operations rather than
 * adding model-specific provider methods.</p>
 */
public final class DiffusionGemmaSelfConditioning {
    private final DiffusionGemmaTextConfig config;
    private final AbstractTensor preNormWeight;
    private final AbstractTensor gateWeight;
    private final AbstractTensor upWeight;
    private final AbstractTensor downWeight;
    private final TensorOperations tensorOperations;
    private final TensorAllocator tensorAllocator;
    private final io.teknek.deliverance.math.WrappedForkJoinPool pool;
    private final MetricRegistry metricRegistry;
    private final TensorRef preNormWeightRef;
    private final TensorRef gateWeightRef;
    private final TensorRef upWeightRef;
    private final TensorRef downWeightRef;
    private final Lighter lighter;
    private final CompositeOps compositeOps;

    public DiffusionGemmaSelfConditioning(DiffusionGemmaTextConfig config, AbstractTensor preNormWeight,
            AbstractTensor gateWeight, AbstractTensor upWeight, AbstractTensor downWeight,
            TensorOperations tensorOperations, TensorAllocator tensorAllocator,
            io.teknek.deliverance.math.WrappedForkJoinPool pool, MetricRegistry metricRegistry) {
        this.config = Objects.requireNonNull(config, "config");
        this.preNormWeight = Objects.requireNonNull(preNormWeight, "preNormWeight");
        this.gateWeight = Objects.requireNonNull(gateWeight, "gateWeight");
        this.upWeight = Objects.requireNonNull(upWeight, "upWeight");
        this.downWeight = Objects.requireNonNull(downWeight, "downWeight");
        this.tensorOperations = Objects.requireNonNull(tensorOperations, "tensorOperations");
        this.tensorAllocator = Objects.requireNonNull(tensorAllocator, "tensorAllocator");
        this.pool = Objects.requireNonNull(pool, "pool");
        this.metricRegistry = Objects.requireNonNull(metricRegistry, "metricRegistry");
        this.preNormWeightRef = null;
        this.gateWeightRef = null;
        this.upWeightRef = null;
        this.downWeightRef = null;
        this.lighter = null;
        this.compositeOps = null;
    }

    /** TensorRef/Lighter implementation of the same block as {@link #forward(AbstractTensor, AbstractTensor)}. */
    public DiffusionGemmaSelfConditioning(DiffusionGemmaTextConfig config, TensorRef preNormWeight,
            TensorRef gateWeight, TensorRef upWeight, TensorRef downWeight, Lighter lighter,
            CompositeOps compositeOps) {
        this.config = Objects.requireNonNull(config, "config");
        this.preNormWeight = null;
        this.gateWeight = null;
        this.upWeight = null;
        this.downWeight = null;
        this.preNormWeightRef = Objects.requireNonNull(preNormWeight, "preNormWeight");
        this.lighter = Objects.requireNonNull(lighter, "lighter");
        this.compositeOps = Objects.requireNonNull(compositeOps, "compositeOps");
        this.gateWeightRef = gateWeight;
        this.upWeightRef = upWeight;
        this.downWeightRef = downWeight;
        this.tensorOperations = null;
        this.tensorAllocator = null;
        this.pool = null;
        this.metricRegistry = null;
    }

    public TensorRef forward(TensorRef inputsEmbeds, TensorRef selfConditioningSignal) {
        validateCanvasEmbeddings(inputsEmbeds, "inputsEmbeds");
        validateCanvasEmbeddings(selfConditioningSignal, "selfConditioningSignal");
        Preconditions.checkArgument(inputsEmbeds.shape().equals(selfConditioningSignal.shape()),
                "inputsEmbeds and selfConditioningSignal shapes must match");
        int batch = inputsEmbeds.shape().dim(0);
        int canvas = inputsEmbeds.shape().dim(1);
        int hidden = config.hiddenSize;
        int rows = batch * canvas;
        try (TensorRef normed = lighter.allocate(DType.F32, inputsEmbeds.shape());
             TensorRef normed2d = lighter.allocate(DType.F32, TensorShape.of(rows, hidden));
             TensorRef gate = lighter.allocate(DType.F32, TensorShape.of(rows, config.intermediateSize));
             TensorRef up = lighter.allocate(DType.F32, TensorShape.of(rows, config.intermediateSize));
             TensorRef down = lighter.allocate(DType.F32, TensorShape.of(rows, hidden));
             TensorRef combined = lighter.allocate(DType.F32, TensorShape.of(rows, hidden))) {
            rmsNorm3d(normed, selfConditioningSignal, preNormWeightRef);
            flatten(normed, normed2d);
            project(normed2d, gate, gateWeightRef, config.intermediateSize, hidden);
            project(normed2d, up, upWeightRef, config.intermediateSize, hidden);
            try (TensorRef activated = compositeOps.activationMultiplyQuantize(new ActivationMultiplyQuantize(
                    gate, up, config.hiddenActivation, DType.F32).offsetAndLength(0, config.intermediateSize))) {
                project(activated, down, downWeightRef, hidden, config.intermediateSize);
            }
            flatten(inputsEmbeds, combined);
            lighter.accumulate(new Accumulate(down).into(combined).offsetAndLength(0, hidden));
            TensorRef output = lighter.allocate(DType.F32, inputsEmbeds.shape());
            try {
                inflate(combined, output, batch, canvas, hidden);
                rmsNorm3d(output, output, null);
                return output;
            } catch (RuntimeException | Error error) {
                output.close();
                throw error;
            }
        }
    }

    private void project(TensorRef input, TensorRef output, TensorRef weight, int outputLength, int inputLength) {
        lighter.dotProductRows(output, input, weight, 0, inputLength, 0, outputLength, 0);
    }

    private void rmsNorm(TensorRef output, TensorRef input, TensorRef weight) {
        int hidden = config.hiddenSize;
        for (int row = 0; row < input.shape().first(); row++) {
            double sum = 0.0;
            for (int column = 0; column < hidden; column++) {
                float value = input.get(row, column);
                sum += value * value;
            }
            float scale = (float) (1.0 / StrictMath.sqrt(sum / hidden + config.rmsNormEps));
            for (int column = 0; column < hidden; column++) {
                output.set(input.get(row, column) * scale * (weight == null ? 1.0f : weight.get(0, column)), row,
                        column);
            }
        }
    }

    private void rmsNorm3d(TensorRef output, TensorRef input, TensorRef weight) {
        int hidden = config.hiddenSize;
        for (int batch = 0; batch < input.shape().dim(0); batch++) {
            for (int position = 0; position < input.shape().dim(1); position++) {
                double sum = 0.0;
                for (int column = 0; column < hidden; column++) {
                    float value = input.get(batch, position, column);
                    sum += value * value;
                }
                float scale = (float) (1.0 / StrictMath.sqrt(sum / hidden + config.rmsNormEps));
                for (int column = 0; column < hidden; column++) {
                    output.set(input.get(batch, position, column) * scale
                            * (weight == null ? 1.0f : weight.get(0, column)), batch, position, column);
                }
            }
        }
    }

    private void flatten(TensorRef source, TensorRef target) {
        int row = 0;
        for (int batch = 0; batch < source.shape().dim(0); batch++) {
            for (int position = 0; position < source.shape().dim(1); position++) {
                for (int column = 0; column < source.shape().dim(2); column++) {
                    target.set(source.get(batch, position, column), row, column);
                }
                row++;
            }
        }
    }

    private void inflate(TensorRef source, TensorRef target, int batch, int canvas, int hidden) {
        int row = 0;
        for (int b = 0; b < batch; b++) {
            for (int position = 0; position < canvas; position++) {
                for (int column = 0; column < hidden; column++) {
                    target.set(source.get(row, column), b, position, column);
                }
                row++;
            }
        }
    }

    public AbstractTensor forward(AbstractTensor inputsEmbeds, AbstractTensor selfConditioningSignal) {
        try (Timer.Context ignored = InferenceProfiler.timer(metricRegistry,
                "diffusiongemma.self_conditioning.forward").time()) {
            validateCanvasEmbeddings(inputsEmbeds, "inputsEmbeds");
            validateCanvasEmbeddings(selfConditioningSignal, "selfConditioningSignal");
            Preconditions.checkArgument(inputsEmbeds.shape().equals(selfConditioningSignal.shape()),
                    "inputsEmbeds and selfConditioningSignal shapes must match");
            int batchSize = (int) inputsEmbeds.shape().dim(0);
            int canvasLength = (int) inputsEmbeds.shape().dim(1);
            int hidden = config.hiddenSize;
            int intermediate = config.intermediateSize;
            int rows = batchSize * canvasLength;

            try (AbstractTensor normed3d = tensorAllocator.getDirty(DType.F32,
                         TensorShape.of(batchSize, canvasLength, hidden));
                 AbstractTensor normed2d = tensorAllocator.getDirty(DType.F32, TensorShape.of(rows, hidden));
                 AbstractTensor gate = tensorAllocator.getDirty(DType.F32, TensorShape.of(rows, intermediate));
                 AbstractTensor up = tensorAllocator.getDirty(DType.F32, TensorShape.of(rows, intermediate));
                 AbstractTensor down = tensorAllocator.getDirty(DType.F32, TensorShape.of(rows, hidden));
                 AbstractTensor combined2d = tensorAllocator.getDirty(DType.F32, TensorShape.of(rows, hidden));
                 AbstractTensor combined3d = tensorAllocator.getDirty(DType.F32,
                         TensorShape.of(batchSize, canvasLength, hidden))) {
                try (Timer.Context ignoredNorm = InferenceProfiler.timer(metricRegistry,
                        "diffusiongemma.self_conditioning.pre_norm").time()) {
                    TensorNormalization.rmsNormLastDim(normed3d, selfConditioningSignal, preNormWeight,
                            config.rmsNormEps, tensorOperations, pool);
                }
                flatten(normed3d, normed2d);
                try (Timer.Context ignoredGateUp = InferenceProfiler.timer(metricRegistry,
                        "diffusiongemma.self_conditioning.gate_up_projection").time()) {
                    AbstractTensor[] results = {gate, up};
                    AbstractTensor[] weights = {gateWeight, upWeight};
                    forChunks(intermediate, (chunkStart, chunkSize) -> tensorOperations.dotProductBatchChunk(results,
                            normed2d, weights, 0, hidden, chunkStart, chunkSize));
                }
                AbstractTensor activated = null;
                try {
                    try (Timer.Context ignoredActivation = InferenceProfiler.timer(metricRegistry,
                            "diffusiongemma.self_conditioning.activation_multiply").time()) {
                        activated = tensorOperations.activationMultiplyQuantize(gate, up, config.hiddenActivation,
                                DType.F32, 0, intermediate);
                    }
                    try (Timer.Context ignoredDown = InferenceProfiler.timer(metricRegistry,
                            "diffusiongemma.self_conditioning.down_projection").time()) {
                        AbstractTensor finalActivated = activated;
                        forChunks(hidden, (chunkStart, chunkSize) -> tensorOperations.dotProductChunk(down,
                                finalActivated, downWeight, 0, intermediate, chunkStart, chunkSize));
                    }
                } finally {
                    if (activated != null) {
                        activated.close();
                    }
                }
                flatten(inputsEmbeds, combined2d);
                tensorOperations.accumulate(combined2d, down, 0, hidden);
                inflate(combined2d, combined3d, batchSize, canvasLength, hidden);

                AbstractTensor output = tensorAllocator.getDirty(DType.F32, TensorShape.of(batchSize, canvasLength, hidden));
                try {
                    try (Timer.Context ignoredPostNorm = InferenceProfiler.timer(metricRegistry,
                            "diffusiongemma.self_conditioning.post_norm").time()) {
                        TensorNormalization.rmsNormLastDim(output, combined3d, null, config.rmsNormEps,
                                tensorOperations, pool);
                    }
                    return output;
                } catch (RuntimeException | Error e) {
                    output.close();
                    throw e;
                }
            }
        }
    }

    private void flatten(AbstractTensor source, AbstractTensor target) {
        int hidden = (int) source.shape().dim(2);
        int rows = (int) (source.shape().dim(0) * source.shape().dim(1));
        if (source.dType() == target.dType()) {
            flattenSameDType(source, target, hidden, rows);
            return;
        }
        try (AbstractTensor flatSource = tensorAllocator.getDirty(source.dType(), TensorShape.of(rows, hidden))) {
            flattenSameDType(source, flatSource, hidden, rows);
            try (AbstractTensor converted = tensorOperations.quantize(flatSource, target.dType(), 0, hidden)) {
                target.copyFrom(converted, 0, 0, (int) converted.size());
            }
        }
    }

    private void flattenSameDType(AbstractTensor source, AbstractTensor target, int hidden, int rows) {
        int canvasLength = (int) source.shape().dim(1);
        TensorPlan flattenPlan = new TensorPlan(tensorOperations, pool).forcedRunMode(TensorPlan.RunMode.CALLER_THREAD);
        flattenPlan.fuseRowsIntStream("diffusiongemma.self_conditioning.flatten", target.shape())
                .read("source", flattenPlan.input("source", source))
                .write("target", flattenPlan.mutable("target", target))
                .map("target[row] = source[batch, position]", TensorPlan.TensorOp.CUSTOM,
                        (ctx, rowOffset, rowLength) -> {
                    int flatRow = (int) rowOffset;
                    int batch = flatRow / canvasLength;
                    int position = flatRow % canvasLength;
                    AbstractTensor src = ctx.tensor("source");
                    AbstractTensor dst = ctx.tensor("target");
                    dst.copyFrom(src, src.getOffset(batch, position, 0), dst.getOffset(flatRow, 0), hidden);
                })
                .tensor()
                .materialize();
    }

    private void inflate(AbstractTensor source, AbstractTensor target, int batchSize, int canvasLength, int hidden) {
        int row = 0;
        for (int batch = 0; batch < batchSize; batch++) {
            for (int position = 0; position < canvasLength; position++) {
                target.copyFrom(source, source.getOffset(row, 0), target.getOffset(batch, position, 0), hidden);
                row++;
            }
        }
    }

    private void forChunks(int length, ChunkConsumer consumer) {
        int split = Math.max(1, tensorOperations.parallelSplitSize());
        for (int start = 0; start < length; start += split) {
            consumer.accept(start, Math.min(split, length - start));
        }
    }

    private void validateCanvasEmbeddings(AbstractTensor tensor, String name) {
        Preconditions.checkArgument(tensor.dims() == 3, name + " must be [batch, canvas, hidden]");
        Preconditions.checkArgument(tensor.shape().dim(2) == config.hiddenSize,
                name + " hidden dimension must match config.hiddenSize");
    }

    private void validateCanvasEmbeddings(TensorRef tensor, String name) {
        Preconditions.checkArgument(tensor.dims() == 3, name + " must be [batch, canvas, hidden]");
        Preconditions.checkArgument(tensor.shape().dim(2) == config.hiddenSize,
                name + " hidden dimension must match config.hiddenSize");
    }

    @FunctionalInterface
    private interface ChunkConsumer {
        void accept(int chunkStart, int chunkSize);
    }
}
