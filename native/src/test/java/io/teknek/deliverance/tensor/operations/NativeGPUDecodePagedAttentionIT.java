package io.teknek.deliverance.tensor.operations;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.Random;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class NativeGPUDecodePagedAttentionIT {

    @Test
    void gpuDecodePagedAttentionMatchesReferenceForSmallF32Pages() {
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new ForkJoinPool(2))) {
            TensorOperations reference = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE,
                    new ArrayQueueTensorAllocator(new MetricRegistry()), pool);
            TensorOperations gpu = loadGpuOperations();
            assumeTrue(gpu != null, "Native GPU operations are not available in this test environment");

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

                assertTrue(gpu.supportsDecodePagedAttention(actual, query, keyPages, valuePages, visibleRows,
                        numberOfHeads, numberOfKeyValueHeads, headSize, 0.25f, null));
                reference.decodePagedAttention(expected, query, keyPages, valuePages, visibleRows, numberOfHeads,
                        numberOfKeyValueHeads, headSize, 0.25f, null);
                gpu.decodePagedAttention(actual, query, keyPages, valuePages, visibleRows, numberOfHeads,
                        numberOfKeyValueHeads, headSize, 0.25f, null);

                assertAllValuesClose(expected, actual, 1.0e-4f, "small");
            }
        }
    }

    @Test
    void gpuDecodePagedAttentionMatchesReferenceForQwenLikeShapePastFailureBoundary() {
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new ForkJoinPool(4))) {
            TensorOperations reference = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE,
                    new ArrayQueueTensorAllocator(new MetricRegistry()), pool);
            TensorOperations gpu = loadGpuOperations();
            assumeTrue(gpu != null, "Native GPU operations are not available in this test environment");

            int numberOfHeads = 32;
            int numberOfKeyValueHeads = 8;
            int headSize = 128;
            int kvLength = numberOfKeyValueHeads * headSize;
            int pageRows = 32;
            int visibleRows = 160;
            int pageCount = (visibleRows + pageRows - 1) / pageRows;
            try (AbstractTensor query = tensor(1, numberOfHeads * headSize, 31);
                 AbstractTensor expected = new FloatBufferTensor(1, numberOfHeads * headSize);
                 AbstractTensor actual = new FloatBufferTensor(1, numberOfHeads * headSize)) {
                AbstractTensor[] keyPages = pages(pageCount, pageRows, kvLength, 43);
                AbstractTensor[] valuePages = pages(pageCount, pageRows, kvLength, 97);
                try {
                    float scale = 1.0f / (float) Math.sqrt(headSize);
                    assertTrue(gpu.supportsDecodePagedAttention(actual, query, keyPages, valuePages, visibleRows,
                            numberOfHeads, numberOfKeyValueHeads, headSize, scale, null));
                    reference.decodePagedAttention(expected, query, keyPages, valuePages, visibleRows, numberOfHeads,
                            numberOfKeyValueHeads, headSize, scale, null);
                    gpu.decodePagedAttention(actual, query, keyPages, valuePages, visibleRows, numberOfHeads,
                            numberOfKeyValueHeads, headSize, scale, null);

                    assertAllValuesClose(expected, actual, 1.0e-3f, "qwen-like visibleRows=" + visibleRows);
                } finally {
                    closeAll(keyPages);
                    closeAll(valuePages);
                }
            }
        }
    }

    @Test
    void gpuDecodePagedAttentionMatchesReferenceForQwenLikeGrowingVisibleRows() {
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new ForkJoinPool(4))) {
            TensorOperations reference = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE,
                    new ArrayQueueTensorAllocator(new MetricRegistry()), pool);
            TensorOperations gpu = loadGpuOperations();
            assumeTrue(gpu != null, "Native GPU operations are not available in this test environment");

            int numberOfHeads = 32;
            int numberOfKeyValueHeads = 8;
            int headSize = 128;
            int kvLength = numberOfKeyValueHeads * headSize;
            int pageRows = 32;
            int maxVisibleRows = 176;
            int pageCount = (maxVisibleRows + pageRows - 1) / pageRows;
            try (AbstractTensor query = tensor(1, numberOfHeads * headSize, 131);
                 AbstractTensor expected = new FloatBufferTensor(1, numberOfHeads * headSize);
                 AbstractTensor actual = new FloatBufferTensor(1, numberOfHeads * headSize)) {
                AbstractTensor[] keyPages = pages(pageCount, pageRows, kvLength, 143);
                AbstractTensor[] valuePages = pages(pageCount, pageRows, kvLength, 197);
                try {
                    float scale = 1.0f / (float) Math.sqrt(headSize);
                    for (int visibleRows : new int[] { 128, 134, 140, 160, 176 }) {
                        fill(query, 131 + visibleRows);
                        expected.clear();
                        actual.clear();
                        assertTrue(gpu.supportsDecodePagedAttention(actual, query, keyPages, valuePages, visibleRows,
                                numberOfHeads, numberOfKeyValueHeads, headSize, scale, null),
                                "visibleRows=" + visibleRows);
                        reference.decodePagedAttention(expected, query, keyPages, valuePages, visibleRows, numberOfHeads,
                                numberOfKeyValueHeads, headSize, scale, null);
                        gpu.decodePagedAttention(actual, query, keyPages, valuePages, visibleRows, numberOfHeads,
                                numberOfKeyValueHeads, headSize, scale, null);
                        assertAllValuesClose(expected, actual, 1.0e-3f, "visibleRows=" + visibleRows);
                    }
                } finally {
                    closeAll(keyPages);
                    closeAll(valuePages);
                }
            }
        }
    }

    @ParameterizedTest
    @CsvSource({
            "32,32,128,128,1,257",
            "32,32,128,128,16,257",
            "32,32,128,128,32,257",
            "32,8,128,128,1,257",
            "32,8,128,128,16,257",
            "32,8,128,128,32,257",
            "32,8,128,128,16,1025"
    })
    void gpuDecodePagedAttentionMatchesReferenceForVllmStyleCases(int numberOfHeads, int numberOfKeyValueHeads,
            int headSize, int valueHeadSize, int pageRows, int visibleRows) {
        assertDecodePagedAttentionMatchesReference("gpu", numberOfHeads, numberOfKeyValueHeads, headSize,
                valueHeadSize, pageRows, visibleRows);
    }

    @ParameterizedTest(name = "provider={0} heads={1} kvHeads={2} d={3} pageRows={5} visibleRows={6}")
    @MethodSource("providerVllmStyleCases")
    void decodePagedAttentionProviderMatchesReferenceForVllmStyleCases(String providerName, int numberOfHeads,
            int numberOfKeyValueHeads, int headSize, int valueHeadSize, int pageRows, int visibleRows) {
        assertDecodePagedAttentionMatchesReference(providerName, numberOfHeads, numberOfKeyValueHeads, headSize,
                valueHeadSize, pageRows, visibleRows);
    }

    private static Stream<org.junit.jupiter.params.provider.Arguments> providerVllmStyleCases() {
        return Stream.of("native-simd", "gpu").flatMap(provider -> Stream.of(
                org.junit.jupiter.params.provider.Arguments.of(provider, 32, 32, 128, 128, 1, 257),
                org.junit.jupiter.params.provider.Arguments.of(provider, 32, 32, 128, 128, 16, 257),
                org.junit.jupiter.params.provider.Arguments.of(provider, 32, 32, 128, 128, 32, 257),
                org.junit.jupiter.params.provider.Arguments.of(provider, 32, 8, 128, 128, 1, 257),
                org.junit.jupiter.params.provider.Arguments.of(provider, 32, 8, 128, 128, 16, 257),
                org.junit.jupiter.params.provider.Arguments.of(provider, 32, 8, 128, 128, 32, 257),
                org.junit.jupiter.params.provider.Arguments.of(provider, 32, 8, 128, 128, 16, 1025)
        ));
    }

    private void assertDecodePagedAttentionMatchesReference(String providerName, int numberOfHeads,
            int numberOfKeyValueHeads, int headSize, int valueHeadSize, int pageRows, int visibleRows) {
        assertEquals(headSize, valueHeadSize, "current GPU path supports D_QK == D_V only");
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new ForkJoinPool(4))) {
            TensorOperations reference = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE,
                    new ArrayQueueTensorAllocator(new MetricRegistry()), pool);
            TensorOperations tested = loadProvider(providerName, pool);
            assumeTrue(tested != null, providerName + " tensor operations are not available in this test environment");

            int kvLength = numberOfKeyValueHeads * headSize;
            int pageCount = (visibleRows + pageRows - 1) / pageRows;
            int physicalPageCount = pageCount + 5;
            try (AbstractTensor query = tensor(1, numberOfHeads * headSize, 313);
                 AbstractTensor expected = new FloatBufferTensor(1, numberOfHeads * valueHeadSize);
                 AbstractTensor actual = new FloatBufferTensor(1, numberOfHeads * valueHeadSize)) {
                AbstractTensor[] physicalKeys = pages(physicalPageCount, pageRows, kvLength, 401);
                AbstractTensor[] physicalValues = pages(physicalPageCount, pageRows, kvLength, 701);
                AbstractTensor[] logicalKeys = new AbstractTensor[pageCount];
                AbstractTensor[] logicalValues = new AbstractTensor[pageCount];
                try {
                    int[] pageTable = deterministicPageTable(pageCount, physicalPageCount, 17);
                    for (int i = 0; i < pageCount; i++) {
                        logicalKeys[i] = physicalKeys[pageTable[i]];
                        logicalValues[i] = physicalValues[pageTable[i]];
                    }
                    float scale = 1.0f / (float) Math.sqrt(headSize);
                    assertTrue(tested.supportsDecodePagedAttention(actual, query, logicalKeys, logicalValues, visibleRows,
                                    numberOfHeads, numberOfKeyValueHeads, headSize, scale, null),
                            "provider=" + providerName + " heads=" + numberOfHeads + " kvHeads="
                                    + numberOfKeyValueHeads + " pageRows=" + pageRows + " visibleRows=" + visibleRows);
                    reference.decodePagedAttention(expected, query, logicalKeys, logicalValues, visibleRows,
                            numberOfHeads, numberOfKeyValueHeads, headSize, scale, null);
                    tested.decodePagedAttention(actual, query, logicalKeys, logicalValues, visibleRows,
                            numberOfHeads, numberOfKeyValueHeads, headSize, scale, null);
                    assertAllValuesClose(expected, actual, 1.0e-3f,
                            "provider=" + providerName + " heads=" + numberOfHeads + " kvHeads="
                                    + numberOfKeyValueHeads + " pageRows=" + pageRows + " visibleRows=" + visibleRows);
                } finally {
                    closeAll(physicalKeys);
                    closeAll(physicalValues);
                }
            }
        }
    }

    private static AbstractTensor tensor(int rows, int cols, int seed) {
        FloatBufferTensor tensor = new FloatBufferTensor(rows, cols);
        fill(tensor, seed);
        return tensor;
    }

    private static AbstractTensor[] pages(int pageCount, int rows, int cols, int seed) {
        AbstractTensor[] pages = new AbstractTensor[pageCount];
        for (int i = 0; i < pageCount; i++) {
            pages[i] = tensor(rows, cols, seed + i * 17);
        }
        return pages;
    }

    private static void fill(AbstractTensor tensor, int seed) {
        for (int row = 0; row < tensor.shape().first(); row++) {
            for (int col = 0; col < tensor.shape().last(); col++) {
                tensor.set((((row * 19 + col * 23 + seed) % 43) - 21) / 21.0f, row, col);
            }
        }
    }

    private static void assertAllValuesClose(AbstractTensor expected, AbstractTensor actual, float tolerance,
            String context) {
        assertEquals(expected.shape(), actual.shape(), context);
        for (int col = 0; col < expected.shape().last(); col++) {
            assertEquals(expected.get(0, col), actual.get(0, col), tolerance, context + " col=" + col);
        }
    }

    private static void closeAll(AbstractTensor[] tensors) {
        for (AbstractTensor tensor : tensors) {
            tensor.close();
        }
    }

    private static int[] deterministicPageTable(int pageCount, int physicalPageCount, int seed) {
        Random random = new Random(seed);
        int[] pageTable = new int[pageCount];
        for (int i = 0; i < pageCount; i++) {
            pageTable[i] = random.nextInt(physicalPageCount);
        }
        return pageTable;
    }

    private static TensorOperations loadGpuOperations() {
        try {
            return new NativeGPUTensorOperations();
        } catch (Throwable t) {
            return null;
        }
    }

    private static TensorOperations loadProvider(String providerName, WrappedForkJoinPool pool) {
        return switch (providerName) {
            case "native-simd" -> new NativeSimdTensorOperations(new PanamaTensorOperations(MachineSpec.VECTOR_TYPE,
                    new ArrayQueueTensorAllocator(new MetricRegistry()), pool));
            case "gpu" -> loadGpuOperations();
            default -> throw new IllegalArgumentException(providerName);
        };
    }
}
