package io.teknek.deliverance.generator2;

import io.teknek.deliverance.tensor2.CompositeOps;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.ScaledSoftMax;
import io.teknek.deliverance.tensor2.TensorRef;
import com.google.common.base.Preconditions;

/** TensorRef-native helpers shared by causal self-attention implementations. */
public abstract class BaseCausalSelfAttention2 implements SelfAttention2 {
    protected final CompositeOps compositeOps;

    protected BaseCausalSelfAttention2(Lighter lighter) {
        this.compositeOps = new CompositeOps(lighter);
    }

    protected void softmax(TensorRef attn, int visibleLength) {
        compositeOps.scaledSoftMax(new ScaledSoftMax(1.0f)
                .target(attn)
                .offsetAndLength(0, visibleLength));
    }

    protected void scaledSoftmax(TensorRef attn, int visibleLength, float scale, Float softcap) {
        compositeOps.scaledSoftMax(new ScaledSoftMax(scale)
                .target(attn)
                .offsetAndLength(0, visibleLength)
                .softcap(softcap));
    }

    /** Packs visible rows from dense page refs into a front-packed tensor. */
    protected int fillVisibleRows(TensorRef packed, TensorRef[] pages, int position, int windowStart,
            int rowWidth) {
        int packedRow = 0;
        int globalOffset = 0;
        for (TensorRef page : pages) {
            int pageRows = Math.min(page.shape().first(), (position + 1) - globalOffset);
            int overlapStart = Math.max(windowStart, globalOffset);
            int overlapEnd = Math.min(position + 1, globalOffset + pageRows);
            if (overlapStart < overlapEnd) {
                int rowOffset = overlapStart - globalOffset;
                int size = overlapEnd - overlapStart;
                for (int row = 0; row < size; row++) {
                    copyRow(page, rowOffset + row, packed, packedRow, rowWidth);
                    packedRow++;
                }
            }
            globalOffset += page.shape().first();
        }
        return packedRow;
    }

    protected int fillVisibleRowsFromDense(TensorRef packed, TensorRef dense, int windowStart,
            int visibleLength, int rowWidth) {
        for (int row = 0; row < visibleLength; row++) {
            copyRow(dense, windowStart + row, packed, row, rowWidth);
        }
        return visibleLength;
    }

    protected void copyRow(TensorRef source, int sourceRow, TensorRef target, int targetRow, int rowWidth) {
        Preconditions.checkArgument(source.dType() == target.dType(), "Row copy requires matching dtypes");
        long bytes = source.dType() == io.teknek.deliverance.DType.Q4
                ? rowWidth / 2L
                : rowWidth * (long) source.dType().size();
        Preconditions.checkArgument(source.dType() != io.teknek.deliverance.DType.Q4 || rowWidth % 2 == 0,
                "Q4 row width must be even");
        long sourceOffset = source.memorySegmentOffset(source.shape().getOffset(sourceRow, 0));
        long targetOffset = target.memorySegmentOffset(target.shape().getOffset(targetRow, 0));
        target.memorySegment().asSlice(targetOffset, bytes)
                .copyFrom(source.memorySegment().asSlice(sourceOffset, bytes));
    }

    protected void closeAll(TensorRef[] tensors) {
        for (TensorRef tensor : tensors) {
            if (tensor != null) {
                tensor.close();
            }
        }
    }
}
