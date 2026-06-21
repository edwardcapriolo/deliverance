package io.teknek.deliverance.generator;

import com.codahale.metrics.MetricRegistry;
import com.codahale.metrics.Timer;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import net.jafama.FastMath;

public abstract class BaseCausalSelfAttention implements SelfAttention {

    protected void applyAttentionSoftcap(AbstractTensor attn, int visibleLength, Float softcap) {
        if (softcap == null) {
            return;
        }
        for (int i = 0; i < visibleLength; i++) {
            float v = attn.get(0, i);
            v /= softcap;
            v = (float) FastMath.tanh(v);
            v *= softcap;
            attn.set(v, 0, i);
        }
    }

    protected void softmax(AbstractTensor attn, int visibleLength) {
        io.teknek.deliverance.tensor.VectorTensorMathUtils.softMax(attn, 0, visibleLength);
    }

    /** Packs visible KV rows from paged cache tensors into a dense front-packed tensor. */
    protected int fillVisibleRows(AbstractTensor packed, AbstractTensor[] pages, int position, int windowStart,
            int rowWidth) {
        int packedRow = 0;
        int globalOffset = 0;
        for (AbstractTensor page : pages) {
            int pageRows = Math.min(page.shape().first(), (position + 1) - globalOffset);
            int overlapStart = Math.max(windowStart, globalOffset);
            int overlapEnd = Math.min(position + 1, globalOffset + pageRows);
            if (overlapStart < overlapEnd) {
                int rowOffset = overlapStart - globalOffset;
                int size = overlapEnd - overlapStart;
                for (int row = 0; row < size; row++) {
                    packed.copyFrom(page, page.getOffset(rowOffset + row, 0), packed.getOffset(packedRow, 0), rowWidth);
                    packedRow++;
                }
            }
            globalOffset += page.shape().first();
        }
        return packedRow;
    }

    protected int fillVisibleRowsFromDense(AbstractTensor packed, AbstractTensor dense, int windowStart,
            int visibleLength, int rowWidth) {
        packed.copyFrom(dense, dense.getOffset(windowStart, 0), packed.getOffset(0, 0), visibleLength * rowWidth);
        return visibleLength;
    }

    protected void copyKvRow(AbstractTensor keyBatch, AbstractTensor valueBatch, int batchIndex,
            AbstractTensor keyTensor, AbstractTensor valueTensor, ConfigurableTensorProvider tensorProvider, int kvLength) {
        try (AbstractTensor keyRow = keyBatch.slice(batchIndex); AbstractTensor valueRow = valueBatch.slice(batchIndex)) {
            if (keyTensor.dType() != keyBatch.dType()) {
                try (AbstractTensor keyQ = tensorProvider.get().quantize(keyRow, keyTensor.dType(), 0, kvLength);
                     AbstractTensor valueQ = tensorProvider.get().quantize(valueRow, valueTensor.dType(), 0, kvLength)) {
                    keyTensor.copyFrom(keyQ, 0, 0, kvLength);
                    valueTensor.copyFrom(valueQ, 0, 0, kvLength);
                }
            } else {
                keyTensor.copyFrom(keyRow, 0, 0, kvLength);
                valueTensor.copyFrom(valueRow, 0, 0, kvLength);
            }
        }
    }

    protected AbstractTensor projectAttentionOutput(AbstractModel model, ConfigurableTensorProvider tensorProvider,
            MetricRegistry metricRegistry, String metricName, AbstractTensor valueOutput,
            AbstractTensor outputProjectionWeights, int inputLength, int outputLength) {
        AbstractTensor result = model.makeDenseTensor(valueOutput.shape().first(), outputLength);
        try (AbstractTensor valueQ = model.maybeQuantize(valueOutput)) {
            try (Timer.Context ignored = InferenceProfiler.timer(metricRegistry, metricName).time()) {
                tensorProvider.get().dotProductChunk(result, valueQ, outputProjectionWeights, 0, inputLength, 0,
                        outputLength);
            }
        }
        return result;
    }

    protected void closeAll(AbstractTensor[] tensors) {
        for (AbstractTensor tensor : tensors) {
            if (tensor != null) {
                tensor.close();
            }
        }
    }
}
