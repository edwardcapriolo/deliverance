package io.teknek.deliverance.generator;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.tensorparallel.SingleRankTensorParallelCollectives;
import io.teknek.deliverance.model.tensorparallel.StaticTensorParallelContext;
import io.teknek.deliverance.model.tensorparallel.TensorParallelCollectives;
import io.teknek.deliverance.model.tensorparallel.TensorParallelContext;
import io.teknek.deliverance.model.tensorparallel.InProcessTensorParallelCollectives;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.KvBufferCache;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.TensorDisplayUtil;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.NaiveTensorOperations;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.List;
import java.time.Duration;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.Future;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.when;

public class CausalSelfAttentionTensorParallelTest {

    @Test
    public void summedRankPartialsMatchFullProductionAttentionOutput() throws Exception {
        Config config = config();
        ConfigurableTensorProvider provider = new ConfigurableTensorProvider(new NaiveTensorOperations());
        TensorAllocator tensorAllocator = new ArrayQueueTensorAllocator(new MetricRegistry());
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new ForkJoinPool(1));
             AbstractTensor input = input();
             AbstractTensor q = identity(4);
             AbstractTensor k = identity(4);
             AbstractTensor v = identity(4);
             AbstractTensor o = identity(4);
             AbstractTensor qRank0 = rowShard(q, 0, 2);
             AbstractTensor kRank0 = rowShard(k, 0, 2);
             AbstractTensor vRank0 = rowShard(v, 0, 2);
             AbstractTensor oRank0 = columnShard(o, 0, 2);
             AbstractTensor qRank1 = rowShard(q, 2, 4);
             AbstractTensor kRank1 = rowShard(k, 2, 4);
             AbstractTensor vRank1 = rowShard(v, 2, 4);
             AbstractTensor oRank1 = columnShard(o, 2, 4)) {

            AbstractTensor full = forward(config, provider, tensorAllocator, pool, new StaticTensorParallelContext(0, 1),
                    new SingleRankTensorParallelCollectives(), input, q, k, v, o);
            InProcessTensorParallelCollectives.Group group = new InProcessTensorParallelCollectives.Group(Duration.ofSeconds(5));
            InProcessTensorParallelCollectives rank0Collectives = new InProcessTensorParallelCollectives(
                    new StaticTensorParallelContext(0, 2), group);
            InProcessTensorParallelCollectives rank1Collectives = new InProcessTensorParallelCollectives(
                    new StaticTensorParallelContext(1, 2), group);

            try (ExecutorService executor = Executors.newFixedThreadPool(2)) {
                Future<AbstractTensor> rank0Future = executor.submit(() -> forward(config, provider, tensorAllocator, pool,
                        new StaticTensorParallelContext(0, 2), rank0Collectives, input, qRank0, kRank0, vRank0, oRank0));
                Future<AbstractTensor> rank1Future = executor.submit(() -> forward(config, provider, tensorAllocator, pool,
                        new StaticTensorParallelContext(1, 2), rank1Collectives, input, qRank1, kRank1, vRank1, oRank1));
                AbstractTensor rank0 = rank0Future.get();
                AbstractTensor rank1 = rank1Future.get();

                try (full; rank0; rank1) {
                    String expected = """
                            [0][0]=  1.0000 [0][1]=  0.0000 [0][2]=  0.0000 [0][3]=  1.0000
                            [1][0]=  0.3302 [1][1]=  0.6698 [1][2]=  0.6698 [1][3]=  0.3302
                            """.trim();
                    assertEquals(expected, normalize(TensorDisplayUtil.pretty2dDisplayAll(full)));
                    assertEquals(expected, normalize(TensorDisplayUtil.pretty2dDisplayAll(rank0)));
                    assertEquals(expected, normalize(TensorDisplayUtil.pretty2dDisplayAll(rank1)));
                }
            }
        }
    }

    private static AbstractTensor forward(Config config, ConfigurableTensorProvider provider,
            TensorAllocator tensorAllocator, WrappedForkJoinPool pool, TensorParallelContext context,
            TensorParallelCollectives collectives, AbstractTensor input, AbstractTensor q, AbstractTensor k,
            AbstractTensor v, AbstractTensor o) {
        AbstractModel model = model(config, tensorAllocator, pool, context, collectives);
        CausalSelfAttention attention = new CausalSelfAttention(model, 0, q, k, v, o, provider, new MetricRegistry());
        KvBufferCache cache = new KvBufferCache(model, new KvBufferCacheSettings(true));
        try (KvBufferCache.KvBuffer kv = cache.getEphemeralKvBuffer()) {
            return attention.forward(new FloatBufferTensor(input), 0, kv, java.util.Optional.empty());
        } finally {
            cache.close();
        }
    }

    private static AbstractModel model(Config config, TensorAllocator tensorAllocator, WrappedForkJoinPool pool,
            TensorParallelContext context, TensorParallelCollectives collectives) {
        AbstractModel model = Mockito.mock(AbstractModel.class);
        when(model.getConfig()).thenReturn(config);
        when(model.getWorkingDType()).thenReturn(DType.F32);
        when(model.getTensorAllocator()).thenReturn(tensorAllocator);
        when(model.getMetricRegistry()).thenReturn(new MetricRegistry());
        when(model.getPool()).thenReturn(pool);
        when(model.getTensorParallelContext()).thenReturn(context);
        when(model.getTensorParallelCollectives()).thenReturn(collectives);
        when(model.getLocalNumberOfHeads()).thenReturn(config.numberOfHeads / context.size());
        when(model.getLocalNumberOfKeyValueHeads()).thenReturn(config.numberOfKeyValueHeads / context.size());
        when(model.getLocalAttentionLength()).thenReturn(config.attentionLength / context.size());
        when(model.getLocalKvLength()).thenReturn(config.kvLength / context.size());
        when(model.makeDenseTensor(Mockito.any(TensorShape.class)))
                .thenAnswer(invocation -> new FloatBufferTensor((TensorShape) invocation.getArgument(0)));
        when(model.makeDenseTensor(Mockito.anyInt(), Mockito.anyInt()))
                .thenAnswer(invocation -> new FloatBufferTensor(
                        (Integer) invocation.getArgument(0),
                        (Integer) invocation.getArgument(1)));
        when(model.maybeQuantize(Mockito.any(AbstractTensor.class)))
                .thenAnswer(invocation -> new FloatBufferTensor((AbstractTensor) invocation.getArgument(0)));
        return model;
    }

    private static Config config() {
        return new Config(16, 4, 8, 2, 2, 1,
                1.0e-6f, 32, 2, List.of(1), ActivationFunction.Type.GELU_PYTORCH_TANH, null, null);
    }

    private static AbstractTensor input() {
        AbstractTensor input = new FloatBufferTensor(2, 4);
        input.set(1.0f, 0, 0);
        input.set(0.0f, 0, 1);
        input.set(0.0f, 0, 2);
        input.set(1.0f, 0, 3);
        input.set(0.0f, 1, 0);
        input.set(1.0f, 1, 1);
        input.set(1.0f, 1, 2);
        input.set(0.0f, 1, 3);
        return input;
    }

    private static AbstractTensor identity(int size) {
        AbstractTensor tensor = new FloatBufferTensor(size, size);
        for (int i = 0; i < size; i++) {
            tensor.set(1.0f, i, i);
        }
        return tensor;
    }

    private static AbstractTensor rowShard(AbstractTensor source, int startInclusive, int endExclusive) {
        int rows = endExclusive - startInclusive;
        int cols = source.shape().last();
        AbstractTensor shard = new FloatBufferTensor(rows, cols);
        for (int row = 0; row < rows; row++) {
            shard.copyFrom(source, source.getOffset(startInclusive + row, 0), shard.getOffset(row, 0), cols);
        }
        return shard;
    }

    private static AbstractTensor columnShard(AbstractTensor source, int startInclusive, int endExclusive) {
        int rows = source.shape().first();
        int cols = endExclusive - startInclusive;
        AbstractTensor shard = new FloatBufferTensor(rows, cols);
        for (int row = 0; row < rows; row++) {
            shard.copyFrom(source, source.getOffset(row, startInclusive), shard.getOffset(row, 0), cols);
        }
        return shard;
    }

    private static String normalize(String display) {
        return display.strip().replaceAll("(?m) +$", "");
    }
}
