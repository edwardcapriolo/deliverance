package io.teknek.deliverance.generator;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator2.PackedBlockAttention2;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.math.VectorMath;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.kv.AttentionPattern;
import io.teknek.deliverance.tensor.kv.CacheExecutionMode;
import io.teknek.deliverance.tensor.kv.KvCacheManager;
import io.teknek.deliverance.tensor.kv.KvCacheSession;
import io.teknek.deliverance.tensor.kv.KvReadView;
import io.teknek.deliverance.tensor.kv.KvWriteCursor;
import io.teknek.deliverance.tensor.operations.NaiveTensorOperations;
import io.teknek.deliverance.tensor.operations.MachineSpec;
import io.teknek.deliverance.tensor.operations.PanamaTensorOperations;
import io.teknek.deliverance.tensor.operations.TensorOperations;
import io.teknek.deliverance.tensor2.CompositeOps;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;
import io.teknek.deliverance.tensor2.TensorRefBackedTensor;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.Optional;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ForkJoinPool;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.when;

class PackedBlockAttention2ParityTest {
    @Test
    void densePrefillMatchesLegacyPackedBlockAttention() {
        MetricRegistry metrics = new MetricRegistry();
        Lighter lighter = new Lighter(metrics);
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new ForkJoinPool(2))) {
            AbstractModel model = model(metrics, lighter, pool);
            PackedBlockAttention legacy = new PackedBlockAttention(model, metrics);
            PackedBlockAttention2 tensorRef = new PackedBlockAttention2(model, metrics);
            int rows = 7;
            int heads = 4;
            int kvHeads = 2;
            int headSize = 8;
            int attentionLength = heads * headSize;
            int kvLength = kvHeads * headSize;
            try (AbstractTensor legacyQuery = tensor(rows, attentionLength, 11);
                 AbstractTensor legacyKeys = tensor(rows, kvLength, 17);
                 AbstractTensor legacyValues = tensor(rows, kvLength, 23);
                 AbstractTensor legacyOutput = new FloatBufferTensor(TensorShape.of(rows, attentionLength));
                 TensorRef query = TensorRef.borrowed(legacyQuery);
                 TensorRef keys = TensorRef.borrowed(legacyKeys);
                 TensorRef values = TensorRef.borrowed(legacyValues);
                 TensorRef output = lighter.allocate(DType.F32, TensorShape.of(rows, attentionLength))) {
                legacy.forward(legacyOutput, legacyQuery, legacyKeys, legacyValues, 0, rows, heads, kvHeads,
                        headSize, 1.0f / (float) Math.sqrt(headSize), null, true);
                tensorRef.forward(output, query, keys, values, 0, rows, heads, kvHeads, headSize,
                        1.0f / (float) Math.sqrt(headSize), null, true);
                assertClose(legacyOutput, new TensorRefBackedTensor(output), 0.001f);
            }
        }
    }

    @Test
    void qwen3FourBDecodeAttentionCharacterization() {
        MetricRegistry metrics = new MetricRegistry();
        Lighter lighter = new Lighter(metrics);
        int heads = 32;
        int kvHeads = 8;
        int headSize = 128;
        int visibleRows = 2048;
        int pageRows = 32;
        int attentionLength = heads * headSize;
        int kvLength = kvHeads * headSize;
        int pageCount = (visibleRows + pageRows - 1) / pageRows;
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new ForkJoinPool(16))) {
            AbstractModel model = parallelModel(metrics, lighter, pool);
            TensorOperations legacyOps = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE,
                    new ArrayQueueTensorAllocator(metrics), pool);
            List<AbstractTensor> legacyKeys = new ArrayList<>();
            List<AbstractTensor> legacyValues = new ArrayList<>();
            List<TensorRef> refKeys = new ArrayList<>();
            List<TensorRef> refValues = new ArrayList<>();
            try (AbstractTensor query = tensor(1, attentionLength, 401);
                 AbstractTensor legacyOutput = new FloatBufferTensor(TensorShape.of(1, attentionLength));
                 TensorRef queryRef = TensorRef.borrowed(query);
                 TensorRef refOutput = lighter.allocate(DType.F32, TensorShape.of(1, attentionLength))) {
                for (int page = 0; page < pageCount; page++) {
                    AbstractTensor key = tensor(pageRows, kvLength, 500 + page);
                    AbstractTensor value = tensor(pageRows, kvLength, 700 + page);
                    legacyKeys.add(key);
                    legacyValues.add(value);
                    refKeys.add(TensorRef.borrowed(key));
                    refValues.add(TensorRef.borrowed(value));
                }
                AbstractTensor[] legacyKeyPages = legacyKeys.toArray(AbstractTensor[]::new);
                AbstractTensor[] legacyValuePages = legacyValues.toArray(AbstractTensor[]::new);
                TensorRef[] refKeyPages = refKeys.toArray(TensorRef[]::new);
                TensorRef[] refValuePages = refValues.toArray(TensorRef[]::new);
                float scale = 1.0f / (float) Math.sqrt(headSize);

                for (int i = 0; i < 5; i++) {
                    legacyOutput.clear();
                    legacyOps.decodePagedAttention(legacyOutput, query, legacyKeyPages, legacyValuePages,
                            visibleRows, heads, kvHeads, headSize, scale, null);
                    lighter.clear(refOutput);
                    new PackedBlockAttention2(model, metrics).decodePagedAttention(refOutput, queryRef, refKeyPages,
                            refValuePages, visibleRows, heads, kvHeads, headSize, scale, null);
                }
                assertClose(legacyOutput, new TensorRefBackedTensor(refOutput), 1.0e-3f);

                int repetitions = 20;
                long legacyStart = System.nanoTime();
                for (int i = 0; i < repetitions; i++) {
                    legacyOutput.clear();
                    legacyOps.decodePagedAttention(legacyOutput, query, legacyKeyPages, legacyValuePages,
                            visibleRows, heads, kvHeads, headSize, scale, null);
                }
                long legacyNanos = System.nanoTime() - legacyStart;

                PackedBlockAttention2 tensorRefAttention = new PackedBlockAttention2(model, metrics);
                long refStart = System.nanoTime();
                for (int i = 0; i < repetitions; i++) {
                    lighter.clear(refOutput);
                    tensorRefAttention.decodePagedAttention(refOutput, queryRef, refKeyPages, refValuePages,
                            visibleRows, heads, kvHeads, headSize, scale, null);
                }
                long refNanos = System.nanoTime() - refStart;
                System.out.printf("[qwen3-4b-decode-attention] visibleRows=%d heads=%d kvHeads=%d headSize=%d "
                                + "legacy_ms=%.3f tensorRef_ms=%.3f legacy_tok_ms=%.3f tensorRef_tok_ms=%.3f%n",
                        visibleRows, heads, kvHeads, headSize, legacyNanos / 1_000_000.0,
                        refNanos / 1_000_000.0, legacyNanos / 1_000_000.0 / repetitions,
                        refNanos / 1_000_000.0 / repetitions);
            } finally {
                refKeys.forEach(TensorRef::close);
                refValues.forEach(TensorRef::close);
                legacyKeys.forEach(AbstractTensor::close);
                legacyValues.forEach(AbstractTensor::close);
            }
        }
    }

    @Test
    void prefixBlockMatchesLegacyPackedBlockAttention() {
        MetricRegistry metrics = new MetricRegistry();
        Lighter lighter = new Lighter(metrics);
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new ForkJoinPool(2))) {
            AbstractModel model = model(metrics, lighter, pool);
            PackedBlockAttention legacy = new PackedBlockAttention(model, metrics);
            PackedBlockAttention2 tensorRef = new PackedBlockAttention2(model, metrics);
            int prefixRows = 5;
            int queryRows = 3;
            int heads = 4;
            int kvHeads = 2;
            int headSize = 8;
            int attentionLength = heads * headSize;
            int kvLength = kvHeads * headSize;
            KvCacheManager manager = new KvCacheManager(1, 32, kvLength, DType.F32,
                    new KvBufferCacheSettings(true), new ArrayQueueTensorAllocator(metrics), metrics);
            try (KvCacheSession session = manager.openSession();
                 AbstractTensor currentKeys = tensor(queryRows, kvLength, 31);
                 AbstractTensor currentValues = tensor(queryRows, kvLength, 37);
                 AbstractTensor query = tensor(queryRows, attentionLength, 41);
                 AbstractTensor legacyOutput = new FloatBufferTensor(TensorShape.of(queryRows, attentionLength));
                 TensorRef queryRef = TensorRef.borrowed(query);
                 TensorRef currentKeysRef = TensorRef.borrowed(currentKeys);
                 TensorRef currentValuesRef = TensorRef.borrowed(currentValues);
                 TensorRef output = lighter.allocate(DType.F32, TensorShape.of(queryRows, attentionLength))) {
                writePrefix(session, prefixRows, kvLength);
                try (KvReadView view = session.readView(0, prefixRows, AttentionPattern.CAUSAL)) {
                    legacy.forward(legacyOutput, query, view, currentKeys, currentValues, prefixRows, queryRows,
                            heads, kvHeads, headSize, 1.0f / (float) Math.sqrt(headSize), null, true);
                }
                try (KvReadView view = session.readView(0, prefixRows, AttentionPattern.CAUSAL)) {
                    tensorRef.forward(output, queryRef, view, currentKeysRef, currentValuesRef, prefixRows, queryRows,
                            heads, kvHeads, headSize, 1.0f / (float) Math.sqrt(headSize), null, true);
                }
                assertClose(legacyOutput, new TensorRefBackedTensor(output), 0.001f);
            }
        }
    }

    @Test
    void cachedPrefixAttentionMatchesFullPrefillFinalRow() {
        MetricRegistry metrics = new MetricRegistry();
        Lighter lighter = new Lighter(metrics);
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new ForkJoinPool(2))) {
            AbstractModel model = model(metrics, lighter, pool);
            PackedBlockAttention2 attention = new PackedBlockAttention2(model, metrics);
            int rows = 5;
            int heads = 2;
            int kvHeads = 1;
            int headSize = 16;
            int attentionLength = heads * headSize;
            int kvLength = kvHeads * headSize;
            try (TensorRef query = tensorRef(lighter, rows, attentionLength, 51);
                 TensorRef keys = tensorRef(lighter, rows, kvLength, 61);
                 TensorRef values = tensorRef(lighter, rows, kvLength, 71);
                 TensorRef fullOutput = lighter.allocate(DType.F32, TensorShape.of(rows, attentionLength));
                 TensorRef cachedOutput = lighter.allocate(DType.F32, TensorShape.of(1, attentionLength));
                 TensorRef queryRow = query.slice(rows - 1)) {
                float scale = 1.0f / (float) Math.sqrt(headSize);
                attention.forward(fullOutput, query, keys, values, 0, rows, heads, kvHeads, headSize, scale, null, true);
                attention.forward(cachedOutput, queryRow, keys, values, rows - 1, 1, heads, kvHeads, headSize,
                        scale, null, true);
                for (int column = 0; column < attentionLength; column++) {
                    assertEquals(fullOutput.get(rows - 1, column), cachedOutput.get(0, column), 1.0e-4f,
                            "column=" + column);
                }
            }
        }
    }

    @ParameterizedTest(name = "prefixRows={0}")
    @ValueSource(ints = {1, 3, 4, 5, 8, 9})
    void tensorRefPagedDecodeMatchesDensePrefixCurrentAttention(int prefixRows) {
        MetricRegistry metrics = new MetricRegistry();
        Lighter lighter = new Lighter(metrics);
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new ForkJoinPool(2))) {
            AbstractModel model = model(metrics, lighter, pool);
            PackedBlockAttention2 attention = new PackedBlockAttention2(model, metrics);
            int queryRows = 1;
            int heads = 4;
            int kvHeads = 2;
            int headSize = 8;
            int attentionLength = heads * headSize;
            int kvLength = kvHeads * headSize;
            KvCacheManager manager = new KvCacheManager(1, 32, kvLength, DType.F32,
                    new KvBufferCacheSettings(true), new ArrayQueueTensorAllocator(metrics), metrics);
            try (KvCacheSession session = manager.openSession();
                 TensorRef query = tensorRef(lighter, queryRows, attentionLength, 41);
                 TensorRef currentKeys = tensorRef(lighter, queryRows, kvLength, 31);
                 TensorRef currentValues = tensorRef(lighter, queryRows, kvLength, 37);
                 TensorRef denseKeys = lighter.allocate(DType.F32, TensorShape.of(prefixRows + queryRows, kvLength));
                 TensorRef denseValues = lighter.allocate(DType.F32, TensorShape.of(prefixRows + queryRows, kvLength));
                 TensorRef denseOutput = lighter.allocate(DType.F32, TensorShape.of(queryRows, attentionLength));
                 TensorRef pagedOutput = lighter.allocate(DType.F32, TensorShape.of(queryRows, attentionLength))) {
                writePrefix(session, prefixRows, kvLength);
                try (KvReadView view = session.readView(0, prefixRows, AttentionPattern.CAUSAL)) {
                    for (int row = 0; row < prefixRows; row++) {
                        try (TensorRef key = view.keyRowRef(row); TensorRef value = view.valueRowRef(row)) {
                            copyRow(key, denseKeys, row, kvLength);
                            copyRow(value, denseValues, row, kvLength);
                        }
                    }
                    copyRow(currentKeys, denseKeys, prefixRows, kvLength);
                    copyRow(currentValues, denseValues, prefixRows, kvLength);
                    attention.forward(denseOutput, query, denseKeys, denseValues, prefixRows, queryRows, heads, kvHeads,
                            headSize, 1.0f / (float) Math.sqrt(headSize), null, true);
                    TensorRef[] keyPages = appendCurrentPage(view.keyPageRefs(), currentKeys);
                    TensorRef[] valuePages = appendCurrentPage(view.valuePageRefs(), currentValues);
                    try {
                        attention.decodePagedAttention(pagedOutput, query, keyPages, valuePages, prefixRows + queryRows,
                                heads, kvHeads, headSize, 1.0f / (float) Math.sqrt(headSize), null);
                    } finally {
                        closePrefixPages(keyPages);
                        closePrefixPages(valuePages);
                    }
                    assertClose(new TensorRefBackedTensor(denseOutput), new TensorRefBackedTensor(pagedOutput), 0.001f);
                }
            }
        }
    }

    private static AbstractModel model(MetricRegistry metrics, Lighter lighter, WrappedForkJoinPool pool) {
        AbstractModel model = Mockito.mock(AbstractModel.class);
        when(model.getMetricRegistry()).thenReturn(metrics);
        when(model.getLighter()).thenReturn(lighter);
        when(model.getCompositeOps()).thenReturn(new CompositeOps(lighter, metrics));
        when(model.primaryTensorOperations()).thenReturn(new NaiveTensorOperations());
        when(model.getPool()).thenReturn(pool);
        when(model.makeDenseTensor(Mockito.any(TensorShape.class)))
                .thenAnswer(invocation -> new FloatBufferTensor((TensorShape) invocation.getArgument(0)));
        when(model.makeDenseTensor(Mockito.anyInt(), Mockito.anyInt()))
                .thenAnswer(invocation -> new FloatBufferTensor(TensorShape.of(
                        (Integer) invocation.getArgument(0), (Integer) invocation.getArgument(1))));
        when(model.makeDenseTensorRef(Mockito.anyInt(), Mockito.anyInt()))
                .thenAnswer(invocation -> lighter.allocate(DType.F32, TensorShape.of(
                        (Integer) invocation.getArgument(0), (Integer) invocation.getArgument(1))));
        Mockito.doAnswer(invocation -> {
            int offset = invocation.getArgument(1);
            int length = invocation.getArgument(2);
            io.teknek.deliverance.math.BiIntConsumer action = invocation.getArgument(5);
            action.accept(offset, length);
            return null;
        }).when(model).runChunks(Mockito.anyString(), Mockito.anyInt(), Mockito.anyInt(), Mockito.anyInt(),
                Mockito.any(Optional.class), Mockito.any(io.teknek.deliverance.math.BiIntConsumer.class));
        return model;
    }

    private static AbstractModel parallelModel(MetricRegistry metrics, Lighter lighter,
            WrappedForkJoinPool pool) {
        AbstractModel model = model(metrics, lighter, pool);
        Mockito.doAnswer(invocation -> {
            int offset = invocation.getArgument(1);
            int length = invocation.getArgument(2);
            int splitSize = invocation.getArgument(3);
            io.teknek.deliverance.math.BiIntConsumer action = invocation.getArgument(5);
            VectorMath.pchunk(offset, length, action, splitSize, pool);
            return null;
        }).when(model).runChunks(Mockito.anyString(), Mockito.anyInt(), Mockito.anyInt(), Mockito.anyInt(),
                Mockito.any(Optional.class), Mockito.any(io.teknek.deliverance.math.BiIntConsumer.class));
        return model;
    }

    private static AbstractTensor tensor(int rows, int columns, int seed) {
        FloatBufferTensor tensor = new FloatBufferTensor(TensorShape.of(rows, columns));
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                tensor.set(((row * 13 + column * 7 + seed) % 41 - 20) / 16.0f, row, column);
            }
        }
        return tensor;
    }

    private static void writePrefix(KvCacheSession session, int prefixRows, int kvLength) {
        try (KvWriteCursor writer = session.writer(CacheExecutionMode.PREFILL_UPDATE_CACHE)) {
            for (int position = 0; position < prefixRows; position++) {
                try (AbstractTensor key = tensor(1, kvLength, 101 + position);
                     AbstractTensor value = tensor(1, kvLength, 201 + position)) {
                    writer.write(0, position, key, value);
                }
            }
            writer.advanceLength(prefixRows);
        }
    }

    private static TensorRef tensorRef(Lighter lighter, int rows, int columns, int seed) {
        TensorRef ref = lighter.allocate(DType.F32, TensorShape.of(rows, columns));
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                ref.set(((row * 13 + column * 7 + seed) % 41 - 20) / 16.0f, row, column);
            }
        }
        return ref;
    }

    private static void copyRow(TensorRef source, TensorRef destination, int destinationRow, int width) {
        for (int column = 0; column < width; column++) {
            destination.set(source.get(0, column), destinationRow, column);
        }
    }

    private static TensorRef[] appendCurrentPage(TensorRef[] prefixPages, TensorRef currentPage) {
        TensorRef[] pages = new TensorRef[prefixPages.length + 1];
        System.arraycopy(prefixPages, 0, pages, 0, prefixPages.length);
        pages[prefixPages.length] = currentPage;
        return pages;
    }

    private static void closePrefixPages(TensorRef[] pages) {
        for (int i = 0; i < pages.length - 1; i++) {
            pages[i].close();
        }
    }

    private static void assertClose(AbstractTensor expected, AbstractTensor actual, float tolerance) {
        assertEquals(expected.shape(), actual.shape());
        for (int row = 0; row < expected.shape().first(); row++) {
            for (int column = 0; column < expected.shape().last(); column++) {
                assertEquals(expected.get(row, column), actual.get(row, column), tolerance,
                        "row=" + row + " column=" + column);
            }
        }
    }
}
