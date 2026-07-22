package io.teknek.deliverance.model.granitemoehybrid;

import io.teknek.deliverance.generator.SelfAttention;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.KvBufferCache;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import net.jafama.FastMath;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.WeakHashMap;
import java.util.function.Consumer;

/** CPU slow-path Mamba2 mixer for GraniteMoeHybrid mamba layers. */
public class GraniteMoeHybridMambaLayer implements SelfAttention {

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
        int sequenceLength = input.shape().first();
        int projectionSize = intermediateSize + convDim + config.mambaNHeads;
        if (startPosition == 0) {
            states.remove(kvMem);
        }
        MambaState state = states.computeIfAbsent(kvMem, ignored -> new MambaState(convDim, config.mambaDConv,
                config.mambaNHeads, config.mambaDHead, config.mambaDState));
        try (AbstractTensor projected = model.makeTensor(sequenceLength, projectionSize);
             AbstractTensor scanOutput = model.makeTensor(sequenceLength, intermediateSize)) {
            tensorProvider.get().batchDotProduct(projected, input, inProjWeights,
                    0, 0, config.embeddingLength, 0, 0, projectionSize);
            for (int row = 0; row < sequenceLength; row++) {
                forwardRow(projected, scanOutput, row, state);
            }
            tensorReducer.ifPresent(func -> func.accept(List.of(scanOutput)));
            AbstractTensor output = model.makeTensor(sequenceLength, config.embeddingLength);
            try (AbstractTensor scanQ = model.maybeQuantize(scanOutput)) {
                tensorProvider.get().batchDotProduct(output, scanQ, outProjWeights,
                        0, 0, intermediateSize, 0, 0, config.embeddingLength);
            }
            return output;
        }
    }

    private void forwardRow(AbstractTensor projected, AbstractTensor scanOutput, int row, MambaState state) {
        float[] gate = new float[intermediateSize];
        float[] convInput = new float[convDim];
        float[] convOutput = new float[convDim];
        float[] dt = new float[config.mambaNHeads];
        for (int i = 0; i < intermediateSize; i++) {
            gate[i] = projected.get(row, i);
        }
        int convOffset = intermediateSize;
        for (int i = 0; i < convDim; i++) {
            convInput[i] = projected.get(row, convOffset + i);
        }
        int dtOffset = intermediateSize + convDim;
        for (int i = 0; i < config.mambaNHeads; i++) {
            dt[i] = softplus(projected.get(row, dtOffset + i) + vectorParam(dtBias, i));
        }
        updateConv(convInput, convOutput, state);
        updateSsmAndWrite(scanOutput, row, gate, convOutput, dt, state);
    }

    private void updateConv(float[] convInput, float[] convOutput, MambaState state) {
        for (int channel = 0; channel < convDim; channel++) {
            System.arraycopy(state.conv[channel], 1, state.conv[channel], 0, config.mambaDConv - 1);
            state.conv[channel][config.mambaDConv - 1] = convInput[channel];
            float value = convBias.isPresent() ? vectorParam(convBias.get(), channel) : 0.0f;
            for (int kernel = 0; kernel < config.mambaDConv; kernel++) {
                value += state.conv[channel][kernel] * convWeight(channel, kernel);
            }
            convOutput[channel] = ActivationFunction.eval(config.activationFunction, value);
        }
    }

    private void updateSsmAndWrite(AbstractTensor scanOutput, int row, float[] gate, float[] convOutput, float[] dt,
            MambaState state) {
        int headsPerGroup = config.mambaNHeads / config.mambaNGroups;
        float sumSquares = 0.0f;
        float[] y = new float[intermediateSize];
        for (int head = 0; head < config.mambaNHeads; head++) {
            int group = head / headsPerGroup;
            int groupOffset = group * config.mambaDState;
            float a = -(float) FastMath.exp(vectorParam(aLog, head));
            float d = vectorParam(dWeights, head);
            for (int dim = 0; dim < config.mambaDHead; dim++) {
                int hiddenIndex = head * config.mambaDHead + dim;
                float x = convOutput[hiddenIndex];
                float value = 0.0f;
                for (int s = 0; s < config.mambaDState; s++) {
                    float b = convOutput[intermediateSize + groupOffset + s];
                    float c = convOutput[intermediateSize + groupsStateSize + groupOffset + s];
                    float recurrent = state.recurrent[head][dim][s]
                            * (float) FastMath.exp(dt[head] * a)
                            + dt[head] * b * x;
                    state.recurrent[head][dim][s] = recurrent;
                    value += recurrent * c;
                }
                value += d * x;
                value *= ActivationFunction.eval(config.activationFunction, gate[hiddenIndex]);
                y[hiddenIndex] = value;
                sumSquares += value * value;
            }
        }
        float invRms = (float) (1.0 / FastMath.sqrt(sumSquares / intermediateSize + config.layerNormEps));
        for (int i = 0; i < intermediateSize; i++) {
            scanOutput.set(y[i] * invRms * vectorParam(normWeights, i), row, i);
        }
    }

    private float convWeight(int channel, int kernel) {
        if (convWeights.dims() == 3) {
            return convWeights.get(channel, 0, kernel);
        }
        return convWeights.get(channel, kernel);
    }

    private static float vectorParam(AbstractTensor tensor, int index) {
        if (tensor.dims() == 1) {
            return tensor.get(index);
        }
        return tensor.get(0, index);
    }

    private static float softplus(float value) {
        if (value > 20.0f) {
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
}
