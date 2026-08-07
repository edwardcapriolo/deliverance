package io.teknek.deliverance.generator;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.operations.MachineSpec;
import io.teknek.deliverance.tensor.operations.PanamaTensorOperations;
import io.teknek.deliverance.tensor.operations.TensorOperations;
import io.teknek.deliverance.tensor.impl.BFloat16BufferTensor;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensorlib.TensorPlan;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.concurrent.ForkJoinPool;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CausalSelfAttentionFlashDecodeTest {

    @ParameterizedTest(name = "{0} gqa partial page")
    @EnumSource(FlashImplementation.class)
    void flashDecodeMatchesStagedAttentionWithGqaAndPartialFinalPage(FlashImplementation implementation) {
        assertFlashMatchesStaged(implementation, 4, 2, 8, 5, 3, 0.25f, 2.0f);
    }

    @ParameterizedTest(name = "{0} single visible row")
    @EnumSource(FlashImplementation.class)
    void flashDecodeMatchesStagedAttentionForSingleVisibleRow(FlashImplementation implementation) {
        assertFlashMatchesStaged(implementation, 2, 2, 4, 1, 2, 0.5f, null);
    }

    @ParameterizedTest(name = "{0} non-f32 pages fallback")
    @EnumSource(value = FlashImplementation.class, names = "SIMD_F32_PARALLEL_HEADS")
    void simdPathFallsBackForNonF32Pages(FlashImplementation implementation) {
        int numberOfHeads = 2;
        int numberOfKeyValueHeads = 1;
        int headSize = 4;
        int attentionLength = numberOfHeads * headSize;
        int kvLength = numberOfKeyValueHeads * headSize;
        int visibleRows = 3;
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new ForkJoinPool(2));
             AbstractTensor query = tensor(1, attentionLength, 3);
             AbstractTensor keyPage = bf16Tensor(3, kvLength, 7);
             AbstractTensor valuePage = bf16Tensor(3, kvLength, 13);
             AbstractTensor expected = new FloatBufferTensor(1, attentionLength);
             AbstractTensor actual = new FloatBufferTensor(1, attentionLength)) {
            AbstractTensor[] keyPages = { keyPage };
            AbstractTensor[] valuePages = { valuePage };

            CausalSelfAttention.flashDecodeAttention(expected, query, keyPages, valuePages, visibleRows, numberOfHeads,
                    numberOfKeyValueHeads, headSize, 0.25f, null);
            implementation.compute(actual, query, keyPages, valuePages, visibleRows, numberOfHeads,
                    numberOfKeyValueHeads, headSize, 0.25f, null, pool);

            for (int col = 0; col < attentionLength; col++) {
                assertEquals(expected.get(0, col), actual.get(0, col), 1.0e-5f, "col=" + col);
            }
        }
    }

    private static void assertFlashMatchesStaged(FlashImplementation implementation, int numberOfHeads,
            int numberOfKeyValueHeads, int headSize, int visibleRows, int pageRows, float scale, Float softcap) {
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new ForkJoinPool(2))) {
            TensorOperations staged = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE,
                    new ArrayQueueTensorAllocator(new MetricRegistry()), pool);
            int attentionLength = numberOfHeads * headSize;
            int kvLength = numberOfKeyValueHeads * headSize;
            try (AbstractTensor query = tensor(1, attentionLength, 3);
                 AbstractTensor keyPage0 = tensor(pageRows, kvLength, 7);
                 AbstractTensor keyPage1 = tensor(pageRows, kvLength, 11);
                 AbstractTensor valuePage0 = tensor(pageRows, kvLength, 13);
                 AbstractTensor valuePage1 = tensor(pageRows, kvLength, 17);
                 AbstractTensor expected = new FloatBufferTensor(1, attentionLength);
                 AbstractTensor actual = new FloatBufferTensor(1, attentionLength)) {
                AbstractTensor[] keyPages = visibleRows <= pageRows
                        ? new AbstractTensor[] { keyPage0 }
                        : new AbstractTensor[] { keyPage0, keyPage1 };
                AbstractTensor[] valuePages = visibleRows <= pageRows
                        ? new AbstractTensor[] { valuePage0 }
                        : new AbstractTensor[] { valuePage0, valuePage1 };

                staged.decodePagedAttention(expected, query, keyPages, valuePages, visibleRows, numberOfHeads,
                        numberOfKeyValueHeads, headSize, scale, softcap);
                implementation.compute(actual, query, keyPages, valuePages, visibleRows, numberOfHeads,
                        numberOfKeyValueHeads, headSize, scale, softcap, pool);

                for (int col = 0; col < attentionLength; col++) {
                    assertEquals(expected.get(0, col), actual.get(0, col), 1.0e-5f, "col=" + col);
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

    private static AbstractTensor bf16Tensor(int rows, int cols, int seed) {
        BFloat16BufferTensor tensor = new BFloat16BufferTensor(rows, cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                tensor.set((((row * 19 + col * 23 + seed) % 43) - 21) / 21.0f, row, col);
            }
        }
        return tensor;
    }

    private enum FlashImplementation {

        SCALAR {
            @Override
            void compute(AbstractTensor valueOut, AbstractTensor query, AbstractTensor[] keyPages,
                    AbstractTensor[] valuePages, int visibleRows, int numberOfHeads, int numberOfKeyValueHeads,
                    int headSize, float scale, Float softcap, WrappedForkJoinPool pool) {
                CausalSelfAttention.flashDecodeAttention(valueOut, query, keyPages, valuePages, visibleRows,
                        numberOfHeads, numberOfKeyValueHeads, headSize, scale, softcap);
            }
        },

        PARALLEL_HEADS {
            @Override
            void compute(AbstractTensor valueOut, AbstractTensor query, AbstractTensor[] keyPages,
                    AbstractTensor[] valuePages, int visibleRows, int numberOfHeads, int numberOfKeyValueHeads,
                    int headSize, float scale, Float softcap, WrappedForkJoinPool pool) {
                CausalSelfAttention.flashDecodeAttentionParallelHeads(valueOut, query, keyPages, valuePages,
                        visibleRows, numberOfHeads, numberOfKeyValueHeads, headSize, scale, softcap, pool);
            }
        },

        SIMD_F32_PARALLEL_HEADS {
            @Override
            void compute(AbstractTensor valueOut, AbstractTensor query, AbstractTensor[] keyPages,
                    AbstractTensor[] valuePages, int visibleRows, int numberOfHeads, int numberOfKeyValueHeads,
                    int headSize, float scale, Float softcap, WrappedForkJoinPool pool) {
                CausalSelfAttention.flashDecodeAttentionSimdF32ParallelHeads(valueOut, query, keyPages, valuePages,
                        visibleRows, numberOfHeads, numberOfKeyValueHeads, headSize, scale, softcap, pool);
            }
        },

        PAGE_BLOCK_SIMD_F32_PARALLEL_HEADS {
            @Override
            void compute(AbstractTensor valueOut, AbstractTensor query, AbstractTensor[] keyPages,
                    AbstractTensor[] valuePages, int visibleRows, int numberOfHeads, int numberOfKeyValueHeads,
                    int headSize, float scale, Float softcap, WrappedForkJoinPool pool) {
                CausalSelfAttention.flashDecodeAttentionPageBlockSimdF32ParallelHeads(valueOut, query, keyPages,
                        valuePages, visibleRows, numberOfHeads, numberOfKeyValueHeads, headSize, scale, softcap, pool);
            }
        },

        TENSOR_PLAN_PARALLEL_HEADS {
            @Override
            void compute(AbstractTensor valueOut, AbstractTensor query, AbstractTensor[] keyPages,
                    AbstractTensor[] valuePages, int visibleRows, int numberOfHeads, int numberOfKeyValueHeads,
                    int headSize, float scale, Float softcap, WrappedForkJoinPool pool) {
                TensorPlan plan = new TensorPlan(new PanamaTensorOperations(MachineSpec.VECTOR_TYPE,
                        new ArrayQueueTensorAllocator(new MetricRegistry()), pool), pool)
                        .forcedRunMode(TensorPlan.RunMode.CALLER_THREAD);
                CausalSelfAttention.flashDecodeAttentionTensorPlanParallelHeads(plan, valueOut, query, keyPages,
                        valuePages, visibleRows, numberOfHeads, numberOfKeyValueHeads, headSize, scale, softcap, pool);
            }
        };

        abstract void compute(AbstractTensor valueOut, AbstractTensor query, AbstractTensor[] keyPages,
                AbstractTensor[] valuePages, int visibleRows, int numberOfHeads, int numberOfKeyValueHeads,
                int headSize, float scale, Float softcap, WrappedForkJoinPool pool);

    }

}
