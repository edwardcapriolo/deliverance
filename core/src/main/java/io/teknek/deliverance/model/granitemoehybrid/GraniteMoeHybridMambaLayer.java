package io.teknek.deliverance.model.granitemoehybrid;

import io.dropwizard.metrics5.Timer;
import io.teknek.deliverance.generator.SelfAttention;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.KvBufferCache;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import net.jafama.FastMath;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.WeakHashMap;
import java.util.function.Consumer;

/** CPU slow-path Mamba2 mixer for GraniteMoeHybrid mamba layers. */
public class GraniteMoeHybridMambaLayer implements SelfAttention {

    private static final String METRIC_FORWARD = "granitemoehybrid.mamba.forward";
    private static final String METRIC_IN_PROJECTION = "granitemoehybrid.mamba.in_projection";
    private static final String METRIC_SCAN = "granitemoehybrid.mamba.scan";
    private static final String METRIC_SCAN_ROW = "granitemoehybrid.mamba.scan.row";
    private static final String METRIC_OUT_PROJECTION = "granitemoehybrid.mamba.out_projection";
    private static final float SOFTPLUS_LINEAR_THRESHOLD = 20.0f;
    private static final VectorSpecies<Float> FLOAT_SPECIES = FloatVector.SPECIES_PREFERRED;

    private final AbstractModel model;
    private final GraniteMoeHybridConfig config;
    private final AbstractTensor inProjWeights;
    private final AbstractTensor convWeights;
    private final Optional<AbstractTensor> convBias;
    private final AbstractTensor dtBias;
    private final AbstractTensor aLog;
    private final AbstractTensor dWeights;
    private final AbstractTensor normWeights;
    private final AbstractTensor outProjWeights;
    private final ConfigurableTensorProvider tensorProvider;
    private final int intermediateSize;
    private final int convDim;
    private final int groupsStateSize;
    private final float[] convBiasValues;
    private final float[][] convWeightValues;
    private final float[] dtBiasValues;
    private final float[] aValues;
    private final float[] dValues;
    private final float[] normValues;
    private final Map<KvBufferCache.KvBuffer, MambaState> states = Collections.synchronizedMap(new WeakHashMap<>());

    public GraniteMoeHybridMambaLayer(AbstractModel model, GraniteMoeHybridConfig config,
            AbstractTensor inProjWeights, AbstractTensor convWeights, Optional<AbstractTensor> convBias,
            AbstractTensor dtBias, AbstractTensor aLog, AbstractTensor dWeights, AbstractTensor normWeights,
            AbstractTensor outProjWeights, ConfigurableTensorProvider tensorProvider) {
        this.model = model;
        this.config = config;
        this.inProjWeights = inProjWeights;
        this.convWeights = convWeights;
        this.convBias = convBias;
        this.dtBias = dtBias;
        this.aLog = aLog;
        this.dWeights = dWeights;
        this.normWeights = normWeights;
        this.outProjWeights = outProjWeights;
        this.tensorProvider = tensorProvider;
        this.intermediateSize = config.mambaExpand * config.embeddingLength;
        this.groupsStateSize = config.mambaNGroups * config.mambaDState;
        this.convDim = intermediateSize + 2 * groupsStateSize;
        this.convBiasValues = loadConvBiasValues(convBias, convDim);
        this.convWeightValues = loadConvWeightValues(convWeights, convDim, config.mambaDConv);
        this.dtBiasValues = loadVectorValues(dtBias, config.mambaNHeads);
        this.aValues = loadAValues(aLog, config.mambaNHeads);
        this.dValues = loadVectorValues(dWeights, config.mambaNHeads);
        this.normValues = loadVectorValues(normWeights, intermediateSize);
        tensorProvider.get().registerModelTensor(inProjWeights);
        tensorProvider.get().registerModelTensor(convWeights);
        convBias.ifPresent(tensorProvider.get()::registerModelTensor);
        tensorProvider.get().registerModelTensor(dtBias);
        tensorProvider.get().registerModelTensor(aLog);
        tensorProvider.get().registerModelTensor(dWeights);
        tensorProvider.get().registerModelTensor(normWeights);
        tensorProvider.get().registerModelTensor(outProjWeights);
    }

    @Override
    public AbstractTensor forward(AbstractTensor input, int startPosition, KvBufferCache.KvBuffer kvMem,
            Optional<Consumer<List<AbstractTensor>>> tensorReducer) {
        try (Timer.Context ignored = InferenceProfiler.timer(model.getMetricRegistry(), METRIC_FORWARD).time()) {
            int sequenceLength = input.shape().first();
            int projectionSize = intermediateSize + convDim + config.mambaNHeads;
            if (startPosition == 0) {
                states.remove(kvMem);
            }
            MambaState state = states.computeIfAbsent(kvMem, ignored2 -> new MambaState(convDim, config.mambaDConv,
                    config.mambaNHeads, config.mambaDHead, config.mambaDState));
            try (AbstractTensor projected = model.makeTensor(sequenceLength, projectionSize);
                 AbstractTensor scanOutput = model.makeTensor(sequenceLength, intermediateSize)) {
                try (Timer.Context ignored2 = InferenceProfiler.timer(model.getMetricRegistry(), METRIC_IN_PROJECTION).time()) {
                    tensorProvider.get().batchDotProduct(projected, input, inProjWeights,
                            0, 0, config.embeddingLength, 0, 0, projectionSize);
                }
                MambaScratch scratch = new MambaScratch(intermediateSize, convDim, config.mambaNHeads);
                try (Timer.Context ignored2 = InferenceProfiler.timer(model.getMetricRegistry(), METRIC_SCAN).time()) {
                    for (int row = 0; row < sequenceLength; row++) {
                        try (Timer.Context ignored3 = InferenceProfiler.timer(model.getMetricRegistry(), METRIC_SCAN_ROW).time()) {
                            forwardRow(projected, scanOutput, row, state, scratch);
                        }
                    }
                }
                tensorReducer.ifPresent(func -> func.accept(List.of(scanOutput)));
                AbstractTensor output = model.makeTensor(sequenceLength, config.embeddingLength);
                try (AbstractTensor scanQ = model.maybeQuantizeReadOnly(scanOutput,
                        "granitemoehybrid.mamba.maybe_quantize.out_projection")) {
                    try (Timer.Context ignored2 = InferenceProfiler.timer(model.getMetricRegistry(), METRIC_OUT_PROJECTION).time()) {
                        tensorProvider.get().batchDotProduct(output, scanQ, outProjWeights,
                                0, 0, intermediateSize, 0, 0, config.embeddingLength);
                    }
                }
                return output;
            }
        }
    }

    private void forwardRow(AbstractTensor projected, AbstractTensor scanOutput, int row, MambaState state,
            MambaScratch scratch) {
        float[] gate = scratch.gate;
        float[] convInput = scratch.convInput;
        float[] convOutput = scratch.convOutput;
        float[] dt = scratch.dt;
        copyRowToArray(projected, row, 0, gate, intermediateSize);
        int convOffset = intermediateSize;
        copyRowToArray(projected, row, convOffset, convInput, convDim);
        int dtOffset = intermediateSize + convDim;
        for (int i = 0; i < config.mambaNHeads; i++) {
            dt[i] = softplus(projected.get(row, dtOffset + i) + dtBiasValues[i]);
        }
        updateConv(convInput, convOutput, state);
        updateSsmAndWrite(scanOutput, row, gate, convOutput, dt, state, scratch.y);
    }

    private void updateConv(float[] convInput, float[] convOutput, MambaState state) {
        for (int channel = 0; channel < convDim; channel++) {
            System.arraycopy(state.conv[channel], 1, state.conv[channel], 0, config.mambaDConv - 1);
            state.conv[channel][config.mambaDConv - 1] = convInput[channel];
            float value = convBiasValues[channel];
            float[] channelState = state.conv[channel];
            float[] channelWeights = convWeightValues[channel];
            for (int kernel = 0; kernel < config.mambaDConv; kernel++) {
                value += channelState[kernel] * channelWeights[kernel];
            }
            convOutput[channel] = ActivationFunction.eval(config.activationFunction, value);
        }
    }

    private void updateSsmAndWrite(AbstractTensor scanOutput, int row, float[] gate, float[] convOutput, float[] dt,
            MambaState state, float[] y) {
        int headsPerGroup = config.mambaNHeads / config.mambaNGroups;
        float sumSquares = 0.0f;
        for (int head = 0; head < config.mambaNHeads; head++) {
            int group = head / headsPerGroup;
            int groupOffset = group * config.mambaDState;
            float d = dValues[head];
            float dtHead = dt[head];
            float decay = (float) FastMath.exp(dtHead * aValues[head]);
            for (int dim = 0; dim < config.mambaDHead; dim++) {
                int hiddenIndex = head * config.mambaDHead + dim;
                float x = convOutput[hiddenIndex];
                float value = 0.0f;
                float[] recurrentState = state.recurrent[head][dim];
                value += updateRecurrentAndDot(recurrentState, convOutput,
                        intermediateSize + groupOffset,
                        intermediateSize + groupsStateSize + groupOffset,
                        decay,
                        dtHead * x,
                        config.mambaDState);
                value += d * x;
                value *= ActivationFunction.eval(config.activationFunction, gate[hiddenIndex]);
                y[hiddenIndex] = value;
                sumSquares += value * value;
            }
        }
        float invRms = (float) (1.0 / FastMath.sqrt(sumSquares / intermediateSize + config.layerNormEps));
        writeNormalizedScanOutput(scanOutput, row, y, invRms);
    }

    private void copyRowToArray(AbstractTensor source, int row, int sourceOffset, float[] dest, int length) {
        if (source instanceof FloatBufferTensor sourceF32) {
            int upperBound = FLOAT_SPECIES.loopBound(length);
            int i = 0;
            for (; i < upperBound; i += FLOAT_SPECIES.length()) {
                sourceF32.getVector(FLOAT_SPECIES, row, sourceOffset + i).intoArray(dest, i);
            }
            for (; i < length; i++) {
                dest[i] = source.get(row, sourceOffset + i);
            }
            return;
        }
        for (int i = 0; i < length; i++) {
            dest[i] = source.get(row, sourceOffset + i);
        }
    }

    private float updateRecurrentAndDot(float[] recurrentState, float[] convOutput, int bOffset, int cOffset,
            float decay, float inputFactor, int stateSize) {
        int upperBound = FLOAT_SPECIES.loopBound(stateSize);
        FloatVector decayVector = FloatVector.broadcast(FLOAT_SPECIES, decay);
        FloatVector factorVector = FloatVector.broadcast(FLOAT_SPECIES, inputFactor);
        FloatVector sumVector = FloatVector.zero(FLOAT_SPECIES);
        int s = 0;
        for (; s < upperBound; s += FLOAT_SPECIES.length()) {
            FloatVector recurrent = FloatVector.fromArray(FLOAT_SPECIES, recurrentState, s)
                    .mul(decayVector)
                    .add(FloatVector.fromArray(FLOAT_SPECIES, convOutput, bOffset + s).mul(factorVector));
            recurrent.intoArray(recurrentState, s);
            sumVector = sumVector.add(recurrent.mul(FloatVector.fromArray(FLOAT_SPECIES, convOutput, cOffset + s)));
        }
        float value = sumVector.reduceLanes(VectorOperators.ADD);
        for (; s < stateSize; s++) {
            float recurrent = recurrentState[s] * decay + convOutput[bOffset + s] * inputFactor;
            recurrentState[s] = recurrent;
            value += recurrent * convOutput[cOffset + s];
        }
        return value;
    }

    private void writeNormalizedScanOutput(AbstractTensor scanOutput, int row, float[] y, float invRms) {
        if (scanOutput instanceof FloatBufferTensor scanF32) {
            FloatVector invRmsVector = FloatVector.broadcast(FLOAT_SPECIES, invRms);
            int upperBound = FLOAT_SPECIES.loopBound(intermediateSize);
            int i = 0;
            for (; i < upperBound; i += FLOAT_SPECIES.length()) {
                FloatVector output = FloatVector.fromArray(FLOAT_SPECIES, y, i)
                        .mul(invRmsVector)
                        .mul(FloatVector.fromArray(FLOAT_SPECIES, normValues, i));
                scanF32.intoTensor(output, row, i);
            }
            for (; i < intermediateSize; i++) {
                scanOutput.set(y[i] * invRms * normValues[i], row, i);
            }
            return;
        }
        for (int i = 0; i < intermediateSize; i++) {
            scanOutput.set(y[i] * invRms * normValues[i], row, i);
        }
    }

    private static float[] loadConvBiasValues(Optional<AbstractTensor> convBias, int convDim) {
        float[] values = new float[convDim];
        if (convBias.isPresent()) {
            AbstractTensor bias = convBias.get();
            for (int i = 0; i < convDim; i++) {
                values[i] = vectorParam(bias, i);
            }
        }
        return values;
    }

    private static float[][] loadConvWeightValues(AbstractTensor convWeights, int convDim, int dConv) {
        float[][] values = new float[convDim][dConv];
        for (int channel = 0; channel < convDim; channel++) {
            for (int kernel = 0; kernel < dConv; kernel++) {
                values[channel][kernel] = convWeights.dims() == 3
                        ? convWeights.get(channel, 0, kernel)
                        : convWeights.get(channel, kernel);
            }
        }
        return values;
    }

    private static float[] loadVectorValues(AbstractTensor tensor, int length) {
        float[] values = new float[length];
        for (int i = 0; i < length; i++) {
            values[i] = vectorParam(tensor, i);
        }
        return values;
    }

    private static float[] loadAValues(AbstractTensor aLog, int length) {
        float[] values = new float[length];
        for (int i = 0; i < length; i++) {
            values[i] = -(float) FastMath.exp(vectorParam(aLog, i));
        }
        return values;
    }

    private static float vectorParam(AbstractTensor tensor, int index) {
        if (tensor.dims() == 1) {
            return tensor.get(index);
        }
        return tensor.get(0, index);
    }

    private static float softplus(float value) {
        // For large positive x, softplus(x) = log(1 + exp(x)) is effectively x and avoids an unnecessary exp.
        if (value > SOFTPLUS_LINEAR_THRESHOLD) {
            return value;
        }
        return (float) FastMath.log1p(FastMath.exp(value));
    }

    private static final class MambaState {
        private final float[][] conv;
        private final float[][][] recurrent;

        private MambaState(int convDim, int convKernel, int heads, int headDim, int stateSize) {
            this.conv = new float[convDim][convKernel];
            this.recurrent = new float[heads][headDim][stateSize];
        }
    }

    private static final class MambaScratch {
        private final float[] gate;
        private final float[] convInput;
        private final float[] convOutput;
        private final float[] dt;
        private final float[] y;

        private MambaScratch(int intermediateSize, int convDim, int heads) {
            this.gate = new float[intermediateSize];
            this.convInput = new float[convDim];
            this.convOutput = new float[convDim];
            this.dt = new float[heads];
            this.y = new float[intermediateSize];
        }
    }
}
