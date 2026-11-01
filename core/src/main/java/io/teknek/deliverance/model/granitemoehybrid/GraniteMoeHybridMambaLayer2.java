package io.teknek.deliverance.model.granitemoehybrid;

import io.teknek.deliverance.generator.ForwardPhase;
import io.teknek.deliverance.generator2.SelfAttention2;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.tensor.KvBufferCache;
import io.teknek.deliverance.tensor.kv.KvCacheSession;
import io.teknek.deliverance.tensor2.BatchDotProduct;
import io.teknek.deliverance.tensor2.TensorRef;
import net.jafama.FastMath;

import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;

/** TensorRef Mamba2 mixer. Its recurrent state is request-local, not KV data. */
final class GraniteMoeHybridMambaLayer2 implements SelfAttention2 {
    private final AbstractModel model;
    private final GraniteMoeHybridConfig config;
    private final int layerIndex;
    private final TensorRef inProj;
    private final TensorRef conv;
    private final TensorRef convBias;
    private final TensorRef dtBias;
    private final TensorRef aLog;
    private final TensorRef d;
    private final TensorRef norm;
    private final TensorRef outProj;
    private final int intermediate;
    private final int groupsState;
    private final int convDim;

    GraniteMoeHybridMambaLayer2(AbstractModel model, GraniteMoeHybridConfig config, int layerIndex,
            TensorRef inProj, TensorRef conv, TensorRef convBias, TensorRef dtBias, TensorRef aLog,
            TensorRef d, TensorRef norm, TensorRef outProj) {
        this.model = model;
        this.config = config;
        this.layerIndex = layerIndex;
        this.inProj = inProj;
        this.conv = conv;
        this.convBias = convBias;
        this.dtBias = dtBias;
        this.aLog = aLog;
        this.d = d;
        this.norm = norm;
        this.outProj = outProj;
        this.intermediate = config.mambaExpand * config.embeddingLength;
        this.groupsState = config.mambaNGroups * config.mambaDState;
        this.convDim = intermediate + 2 * groupsState;
    }

    @Override
    public TensorRef forward(TensorRef input, int startPosition, KvCacheSession session,
            Optional<Consumer<List<TensorRef>>> reducer, ForwardPhase phase, int batchSize,
            int sequenceLength, int[] attentionMask) {
        if (batchSize != 1 || input.shape().first() != sequenceLength) {
            throw new IllegalArgumentException("Granite Mamba2 currently requires a single flattened sequence");
        }
        MambaSessionState states = session.getOrAttach(MambaSessionState.class,
                () -> new MambaSessionState(config.numberOfLayers));
        MambaState state = states.state(layerIndex, convDim, config);
        if (startPosition == 0) state.clear();

        int rows = (int) input.shape().first();
        int projection = intermediate + convDim + config.mambaNHeads;
        TensorRef projected = model.makeDenseTensorRef(rows, projection);
        TensorRef scan = model.makeDenseTensorRef(rows, intermediate);
        try {
            model.getLighter().batchDotProduct(new BatchDotProduct().result(projected).a(input).b(inProj)
                    .aColumnOffset(0).bColumnOffset(0).columnLength(config.embeddingLength)
                    .resultRowOffset(0).bRowOffset(0).rowChunkSize(projection));
            Scratch scratch = new Scratch(intermediate, convDim, config.mambaNHeads);
            for (int row = 0; row < rows; row++) forwardRow(projected, scan, row, state, scratch);
            reducer.ifPresent(fn -> fn.accept(List.of(scan)));
            TensorRef output = model.makeDenseTensorRef(rows, config.embeddingLength);
            try {
                model.getLighter().batchDotProduct(new BatchDotProduct().result(output).a(scan).b(outProj)
                        .aColumnOffset(0).bColumnOffset(0).columnLength(intermediate)
                        .resultRowOffset(0).bRowOffset(0).rowChunkSize(config.embeddingLength));
                return output;
            } catch (RuntimeException | Error e) {
                output.close();
                throw e;
            }
        } finally {
            projected.close();
            scan.close();
        }
    }

    @Override
    public TensorRef forward(TensorRef input, int startPosition, KvBufferCache.KvBuffer kvMem,
            Optional<Consumer<List<TensorRef>>> reducer) {
        throw new UnsupportedOperationException("Granite Mamba2 requires KVCache2 session");
    }

    private void forwardRow(TensorRef projected, TensorRef scan, int row, MambaState state, Scratch s) {
        copy(projected, row, 0, s.gate, intermediate);
        copy(projected, row, intermediate, s.convInput, convDim);
        int dtOffset = intermediate + convDim;
        for (int head = 0; head < config.mambaNHeads; head++) {
            s.dt[head] = softplus(projected.get(row, dtOffset + head) + value(dtBias, head));
        }
        for (int channel = 0; channel < convDim; channel++) {
            System.arraycopy(state.conv[channel], 1, state.conv[channel], 0, config.mambaDConv - 1);
            state.conv[channel][config.mambaDConv - 1] = s.convInput[channel];
            float value = convBias == null ? 0.0f : value(convBias, channel);
            for (int kernel = 0; kernel < config.mambaDConv; kernel++) {
                value += state.conv[channel][kernel] * value(conv, channel, kernel);
            }
            s.convOutput[channel] = ActivationFunction.eval(config.activationFunction, value);
        }
        int headsPerGroup = config.mambaNHeads / config.mambaNGroups;
        float sumSquares = 0.0f;
        for (int head = 0; head < config.mambaNHeads; head++) {
            int groupOffset = (head / headsPerGroup) * config.mambaDState;
            float decay = (float) FastMath.exp(s.dt[head] * -((float) FastMath.exp(value(aLog, head))));
            for (int dim = 0; dim < config.mambaDHead; dim++) {
                int hidden = head * config.mambaDHead + dim;
                float[] recurrent = state.recurrent[head][dim];
                float result = 0.0f;
                for (int n = 0; n < config.mambaDState; n++) {
                    recurrent[n] = recurrent[n] * decay + s.convOutput[intermediate + groupOffset + n]
                            * s.dt[head] * s.convOutput[hidden];
                    result += recurrent[n] * s.convOutput[intermediate + groupsState + groupOffset + n];
                }
                float y = (result + value(d, head) * s.convOutput[hidden])
                        * ActivationFunction.eval(config.activationFunction, s.gate[hidden]);
                s.y[hidden] = y;
                sumSquares += y * y;
            }
        }
        float scale = (float) (1.0 / FastMath.sqrt(sumSquares / intermediate + config.layerNormEps));
        for (int i = 0; i < intermediate; i++) scan.set(s.y[i] * scale * value(norm, i), row, i);
    }

    private static void copy(TensorRef source, int row, int offset, float[] target, int length) {
        for (int i = 0; i < length; i++) target[i] = source.get(row, offset + i);
    }

    private static float value(TensorRef tensor, int index) {
        return tensor.dims() == 1 ? tensor.get(index) : tensor.get(0, index);
    }

    private static float value(TensorRef tensor, int row, int column) {
        return tensor.dims() == 3 ? tensor.get(row, 0, column) : tensor.get(row, column);
    }

    private static float softplus(float x) {
        return x > 20.0f ? x : (float) FastMath.log1p(FastMath.exp(x));
    }

    private static final class MambaSessionState implements AutoCloseable {
        private final MambaState[] states;
        private MambaSessionState(int layers) { states = new MambaState[layers]; }
        private synchronized MambaState state(int layer, int convDim, GraniteMoeHybridConfig config) {
            if (states[layer] == null) states[layer] = new MambaState(convDim, config);
            return states[layer];
        }
        @Override public void close() { }
    }

    private static final class MambaState {
        private final float[][] conv;
        private final float[][][] recurrent;
        private MambaState(int convDim, GraniteMoeHybridConfig c) {
            conv = new float[convDim][c.mambaDConv];
            recurrent = new float[c.mambaNHeads][c.mambaDHead][c.mambaDState];
        }
        private void clear() {
            for (float[] row : conv) java.util.Arrays.fill(row, 0.0f);
            for (float[][] head : recurrent) for (float[] row : head) java.util.Arrays.fill(row, 0.0f);
        }
    }

    private static final class Scratch {
        private final float[] gate, convInput, convOutput, dt, y;
        private Scratch(int intermediate, int convDim, int heads) {
            gate = new float[intermediate]; convInput = new float[convDim]; convOutput = new float[convDim];
            dt = new float[heads]; y = new float[intermediate];
        }
    }
}
