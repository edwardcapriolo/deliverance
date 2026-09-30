package io.teknek.deliverance.generator2;

import io.teknek.deliverance.generator.ForwardPhase;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.kv.KvCacheSession;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.CompositeOps;
import io.teknek.deliverance.tensor2.ScaledSoftMax;
import io.teknek.deliverance.tensor2.TensorRef;

import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;

/** TensorRef BERT self-attention with bidirectional masking. */
public final class BertSelfAttention2 implements SelfAttention2 {
    private final AbstractModel model;
    private final int attentionLength;
    private final int numberOfHeads;
    private final int headSize;
    private final float attentionScale;
    private final TensorRef queryWeight;
    private final TensorRef keyWeight;
    private final TensorRef valueWeight;
    private final TensorRef outputWeight;
    private final TensorRef queryBias;
    private final TensorRef keyBias;
    private final TensorRef valueBias;
    private final TensorRef outputBias;
    private final Lighter lighter;
    private final CompositeOps compositeOps;

    public BertSelfAttention2(AbstractModel model, TensorRef queryWeight, TensorRef keyWeight, TensorRef valueWeight,
            TensorRef outputWeight, TensorRef queryBias, TensorRef keyBias, TensorRef valueBias, TensorRef outputBias,
            Lighter lighter) {
        this.model = model;
        this.attentionLength = model.getConfig().attentionLength;
        this.numberOfHeads = model.getConfig().numberOfHeads;
        this.headSize = model.getConfig().headSize;
        this.attentionScale = model.getConfig().attentionMultiplier != null
                ? model.getConfig().attentionMultiplier
                : (float) (1.0 / StrictMath.sqrt(headSize));
        this.queryWeight = queryWeight;
        this.keyWeight = keyWeight;
        this.valueWeight = valueWeight;
        this.outputWeight = outputWeight;
        this.queryBias = queryBias;
        this.keyBias = keyBias;
        this.valueBias = valueBias;
        this.outputBias = outputBias;
        this.lighter = lighter;
        this.compositeOps = new CompositeOps(lighter, model.getMetricRegistry());
    }

    @Override
    public TensorRef forward(TensorRef input, int startPosition, KvCacheSession kvSession,
            Optional<Consumer<List<TensorRef>>> tensorReducer, ForwardPhase phase, int batchSize,
            int sequenceLength, int[] attentionMask) {
        if (input.shape().first() != (long) batchSize * sequenceLength) {
            throw new IllegalArgumentException("BERT attention input does not match batch and sequence shape");
        }
        if (attentionMask != null && attentionMask.length != batchSize * sequenceLength) {
            throw new IllegalArgumentException("BERT attention mask does not match batch and sequence shape");
        }
        TensorRef query = model.makeDenseTensorRef(input.shape());
        TensorRef key = model.makeDenseTensorRef(input.shape());
        TensorRef value = model.makeDenseTensorRef(input.shape());
        TensorRef attended = model.makeDenseTensorRef(input.shape());
        try {
            project(query, input, queryWeight, queryBias);
            project(key, input, keyWeight, keyBias);
            project(value, input, valueWeight, valueBias);
            for (int batch = 0; batch < batchSize; batch++) {
                int rowStart = batch * sequenceLength;
                for (int queryRow = 0; queryRow < sequenceLength; queryRow++) {
                    try (TensorRef queryVector = query.slice(rowStart + queryRow)) {
                        for (int head = 0; head < numberOfHeads; head++) {
                            int headOffset = head * headSize;
                            try (TensorRef scores = lighter.allocate(query.dType(),
                                    TensorShape.of(1, sequenceLength))) {
                                lighter.dotProductRows(scores, queryVector, key, headOffset, headOffset,
                                        headSize, rowStart, sequenceLength, 0);
                                mask(scores, batch, sequenceLength, attentionMask);
                                compositeOps.scaledSoftMax(new ScaledSoftMax(attentionScale)
                                        .target(scores).offsetAndLength(0, sequenceLength));
                                for (int column = 0; column < headSize; column++) {
                                    float sum = 0.0f;
                                    for (int keyRow = 0; keyRow < sequenceLength; keyRow++) {
                                        sum += scores.get(0, keyRow) * value.get(rowStart + keyRow, headOffset + column);
                                    }
                                    attended.set(sum, rowStart + queryRow, headOffset + column);
                                }
                            }
                        }
                    }
                }
            }
            TensorRef output = model.makeDenseTensorRef(input.shape());
            try {
                project(output, attended, outputWeight, outputBias);
                tensorReducer.ifPresent(func -> func.accept(List.of(output)));
                return output;
            } catch (RuntimeException | Error e) {
                output.close();
                throw e;
            }
        } finally {
            query.close();
            key.close();
            value.close();
            attended.close();
        }
    }

    private void project(TensorRef output, TensorRef input, TensorRef weight, TensorRef bias) {
        int outputLength = (int) output.shape().last();
        model.runChunks("bertselfattention2.projection", 0, outputLength,
                model.primaryTensorOperations().parallelSplitSize(), Optional.empty(),
                (chunkStart, chunkSize) -> lighter.dotProductRows(output, input, weight, 0,
                        attentionLength, chunkStart, chunkSize, chunkStart));
        addBias(output, bias);
    }

    private void addBias(TensorRef target, TensorRef bias) {
        for (int row = 0; row < target.shape().first(); row++) {
            for (int column = 0; column < target.shape().last(); column++) {
                target.set(target.get(row, column) + bias.get(0, column), row, column);
            }
        }
    }

    private void mask(TensorRef scores, int batch, int sequenceLength, int[] attentionMask) {
        if (attentionMask == null) {
            return;
        }
        for (int keyRow = 0; keyRow < sequenceLength; keyRow++) {
            if (attentionMask[batch * sequenceLength + keyRow] == 0) {
                scores.set(Float.NEGATIVE_INFINITY, 0, keyRow);
            }
        }
    }

    @Override
    public TensorRef forward(TensorRef input, int startPosition, io.teknek.deliverance.tensor.KvBufferCache.KvBuffer kvMem,
            Optional<Consumer<List<TensorRef>>> tensorReducer) {
        throw new UnsupportedOperationException("BERT TensorRef attention requires batch and mask metadata");
    }
}
