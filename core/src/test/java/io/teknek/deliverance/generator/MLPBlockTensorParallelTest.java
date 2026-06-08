package io.teknek.deliverance.generator;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.tensorparallel.InProcessTensorParallelCollectives;
import io.teknek.deliverance.model.tensorparallel.SingleRankTensorParallelCollectives;
import io.teknek.deliverance.model.tensorparallel.StaticTensorParallelContext;
import io.teknek.deliverance.model.tensorparallel.TensorParallelCollectives;
import io.teknek.deliverance.model.tensorparallel.TensorParallelContext;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.TensorDisplayUtil;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.NaiveTensorOperations;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.time.Duration;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.Future;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.when;

public class MLPBlockTensorParallelTest {

    @Test
    public void productionMlpBlockRankOutputsMatchFullMlpOutput() throws Exception {
        Config config = config();
        ConfigurableTensorProvider provider = new ConfigurableTensorProvider(new NaiveTensorOperations());
        TensorAllocator tensorAllocator = new ArrayQueueTensorAllocator(new MetricRegistry());
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new ForkJoinPool(1));
             AbstractTensor input = input();
             AbstractTensor gate = matrix(4, 3, 0.10f);
             AbstractTensor up = matrix(4, 3, 0.20f);
             AbstractTensor down = matrix(3, 4, -0.15f);
             AbstractTensor gateRank0 = rowShard(gate, 0, 2);
             AbstractTensor upRank0 = rowShard(up, 0, 2);
             AbstractTensor downRank0 = columnShard(down, 0, 2);
             AbstractTensor gateRank1 = rowShard(gate, 2, 4);
             AbstractTensor upRank1 = rowShard(up, 2, 4);
             AbstractTensor downRank1 = columnShard(down, 2, 4)) {

            AbstractTensor full = mlpBlock(config, tensorAllocator, pool, new StaticTensorParallelContext(0, 1),
                    new SingleRankTensorParallelCollectives(), provider, gate, down, up, null).forward(input,
                    Optional.empty());

            InProcessTensorParallelCollectives.Group group = new InProcessTensorParallelCollectives.Group(Duration.ofSeconds(5));
            InProcessTensorParallelCollectives rank0Collectives = new InProcessTensorParallelCollectives(
                    new StaticTensorParallelContext(0, 2), group);
            InProcessTensorParallelCollectives rank1Collectives = new InProcessTensorParallelCollectives(
                    new StaticTensorParallelContext(1, 2), group);

            try (ExecutorService executor = Executors.newFixedThreadPool(2)) {
                Future<AbstractTensor> rank0Future = executor.submit(() -> mlpBlock(config, tensorAllocator, pool,
                        new StaticTensorParallelContext(0, 2), rank0Collectives, provider, gateRank0, downRank0,
                        upRank0, "layer.0.mlp.down_proj").forward(input, Optional.empty()));
                Future<AbstractTensor> rank1Future = executor.submit(() -> mlpBlock(config, tensorAllocator, pool,
                        new StaticTensorParallelContext(1, 2), rank1Collectives, provider, gateRank1, downRank1,
                        upRank1, "layer.0.mlp.down_proj").forward(input, Optional.empty()));

                AbstractTensor rank0 = rank0Future.get();
                AbstractTensor rank1 = rank1Future.get();
                try (full; rank0; rank1) {
                    String expected = """
                            [0][0]= -5.0217 [0][1]=-10.9343 [0][2]=-16.8470
                            """.trim();
                    assertEquals(expected, normalize(TensorDisplayUtil.pretty2dDisplayAll(full)));
                    assertEquals(expected, normalize(TensorDisplayUtil.pretty2dDisplayAll(rank0)));
                    assertEquals(expected, normalize(TensorDisplayUtil.pretty2dDisplayAll(rank1)));
                }
            }
        }
    }

    private static MLPBlock mlpBlock(Config config, TensorAllocator tensorAllocator, WrappedForkJoinPool pool,
            TensorParallelContext context, TensorParallelCollectives collectives, ConfigurableTensorProvider provider,
            AbstractTensor gate, AbstractTensor down, AbstractTensor up, String collectiveKey) {
        return new MLPBlock(model(config, tensorAllocator, pool, context, collectives), ActivationFunction.Type.SILU,
                gate, down, up, provider, collectiveKey);
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
        when(model.makeTensor(Mockito.anyInt(), Mockito.anyInt()))
                .thenAnswer(invocation -> new FloatBufferTensor((Integer) invocation.getArgument(0),
                        (Integer) invocation.getArgument(1)));
        when(model.makeDenseTensor(Mockito.any(TensorShape.class)))
                .thenAnswer(invocation -> new FloatBufferTensor((TensorShape) invocation.getArgument(0)));
        when(model.maybeQuantize(Mockito.any(AbstractTensor.class)))
                .thenAnswer(invocation -> new FloatBufferTensor((AbstractTensor) invocation.getArgument(0)));
        return model;
    }

    private static Config config() {
        return new Config(16, 3, 4, 1, 1, 1,
                1.0e-6f, 32, 2, List.of(1), ActivationFunction.Type.SILU, null, null);
    }

    private static AbstractTensor input() {
        AbstractTensor input = new FloatBufferTensor(1, 3);
        input.set(0.5f, 0, 0);
        input.set(-1.0f, 0, 1);
        input.set(2.0f, 0, 2);
        return input;
    }

    private static AbstractTensor matrix(int rows, int cols, float scale) {
        AbstractTensor tensor = new FloatBufferTensor(rows, cols);
        int value = 1;
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                tensor.set(value++ * scale, row, col);
            }
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
