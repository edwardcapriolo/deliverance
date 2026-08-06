package io.teknek.deliverance.tensor.operations;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.VectorTensorMathUtils;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.api.Test;

import java.util.concurrent.ForkJoinPool;

import static org.junit.jupiter.api.Assertions.assertEquals;

class TensorOperationsDecodePagedAttentionTest {

    @Test
    void decodePagedAttentionMatchesExplicitPageLoopWithGqaAndPartialFinalPage() {
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new ForkJoinPool(2))) {
            TensorOperations ops = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE,
                    new ArrayQueueTensorAllocator(new MetricRegistry()), pool);
            int numberOfHeads = 4;
            int numberOfKeyValueHeads = 2;
            int headSize = 8;
            int kvLength = numberOfKeyValueHeads * headSize;
            int visibleRows = 5;
            try (AbstractTensor query = tensor(1, numberOfHeads * headSize, 3);
                 AbstractTensor keyPage0 = tensor(3, kvLength, 7);
                 AbstractTensor keyPage1 = tensor(3, kvLength, 11);
                 AbstractTensor valuePage0 = tensor(3, kvLength, 13);
                 AbstractTensor valuePage1 = tensor(3, kvLength, 17);
                 AbstractTensor expected = new FloatBufferTensor(1, numberOfHeads * headSize);
                 AbstractTensor actual = new FloatBufferTensor(1, numberOfHeads * headSize)) {
                AbstractTensor[] keyPages = { keyPage0, keyPage1 };
                AbstractTensor[] valuePages = { valuePage0, valuePage1 };

                explicitPageLoop(expected, query, keyPages, valuePages, visibleRows, numberOfHeads,
                        numberOfKeyValueHeads, headSize, 0.25f, 2.0f);
                ops.decodePagedAttention(actual, query, keyPages, valuePages, visibleRows, numberOfHeads,
                        numberOfKeyValueHeads, headSize, 0.25f, 2.0f);

                for (int col = 0; col < expected.shape().last(); col++) {
                    assertEquals(expected.get(0, col), actual.get(0, col), 1.0e-5f, "col=" + col);
                }
            }
        }
    }

    private static void explicitPageLoop(AbstractTensor out, AbstractTensor query, AbstractTensor[] keyPages,
            AbstractTensor[] valuePages, int visibleRows, int numberOfHeads, int numberOfKeyValueHeads, int headSize,
            float scale, Float softcap) {
        int headGroupSize = numberOfHeads / numberOfKeyValueHeads;
        try (AbstractTensor attn = new FloatBufferTensor(1, visibleRows)) {
            for (int head = 0; head < numberOfHeads; head++) {
                int kvHead = head / headGroupSize;
                int xoffset = kvHead * headSize;
                int yoffset = head * headSize;
                int globalOffset = 0;
                for (int pageIndex = 0; pageIndex < keyPages.length && globalOffset < visibleRows; pageIndex++) {
                    AbstractTensor keyPage = keyPages[pageIndex];
                    int rows = (int) Math.min(keyPage.shape().first(), visibleRows - globalOffset);
                    for (int row = 0; row < rows; row++) {
                        float score = 0.0f;
                        for (int col = 0; col < headSize; col++) {
                            score += query.get(0, yoffset + col) * keyPage.get(row, xoffset + col);
                        }
                        attn.set(score, 0, globalOffset + row);
                    }
                    globalOffset += rows;
                }
                VectorTensorMathUtils.scaledSoftMax(attn, 0, visibleRows, scale, softcap);
                for (int col = 0; col < headSize; col++) {
                    out.set(0.0f, 0, yoffset + col);
                }
                globalOffset = 0;
                for (int pageIndex = 0; pageIndex < valuePages.length && globalOffset < visibleRows; pageIndex++) {
                    AbstractTensor valuePage = valuePages[pageIndex];
                    int rows = (int) Math.min(valuePage.shape().first(), visibleRows - globalOffset);
                    for (int row = 0; row < rows; row++) {
                        float weight = attn.get(0, globalOffset + row);
                        for (int col = 0; col < headSize; col++) {
                            out.set(out.get(0, yoffset + col) + weight * valuePage.get(row, xoffset + col), 0,
                                    yoffset + col);
                        }
                    }
                    globalOffset += rows;
                }
            }
        }
    }

    private static AbstractTensor tensor(int rows, int cols, int seed) {
        FloatBufferTensor tensor = new FloatBufferTensor(rows, cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                tensor.set((((row * 19 + col * 23 + seed) % 43) - 21) / 21.0f, row, col);
            }
        }
        return tensor;
    }
}
