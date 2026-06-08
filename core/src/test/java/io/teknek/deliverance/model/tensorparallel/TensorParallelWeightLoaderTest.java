package io.teknek.deliverance.model.tensorparallel;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.safetensors.TensorShardAxis;
import io.teknek.deliverance.safetensors.TensorShardSpec;
import io.teknek.deliverance.safetensors.WeightLoader;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorInfo;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

public class TensorParallelWeightLoaderTest {

    @Test
    public void queryProjectionUsesAttentionRowShard() {
        RecordingWeightLoader delegate = new RecordingWeightLoader();
        TensorParallelWeightLoader loader = loader(delegate, new StaticTensorParallelContext(1, 4));

        loader.load("model.layers.0.self_attn.q_proj.weight").close();

        assertEquals(new TensorShardSpec(TensorShardAxis.ROWS, 512, 1024), delegate.lastShardSpec);
    }

    @Test
    public void keyAndValueProjectionsUseKvRowShard() {
        RecordingWeightLoader delegate = new RecordingWeightLoader();
        TensorParallelWeightLoader loader = loader(delegate, new StaticTensorParallelContext(2, 4));

        loader.load("model.layers.0.self_attn.k_proj.weight").close();
        assertEquals(new TensorShardSpec(TensorShardAxis.ROWS, 512, 768), delegate.lastShardSpec);

        loader.load("model.layers.0.self_attn.v_proj.weight").close();
        assertEquals(new TensorShardSpec(TensorShardAxis.ROWS, 512, 768), delegate.lastShardSpec);
    }

    @Test
    public void attentionOutputProjectionUsesAttentionColumnShard() {
        RecordingWeightLoader delegate = new RecordingWeightLoader();
        TensorParallelWeightLoader loader = loader(delegate, new StaticTensorParallelContext(3, 4));

        loader.load("model.layers.0.self_attn.o_proj.weight").close();

        assertEquals(new TensorShardSpec(TensorShardAxis.COLUMNS, 1536, 2048), delegate.lastShardSpec);
    }

    @Test
    public void mlpInputAndOutputProjectionsUseMlpRanges() {
        RecordingWeightLoader delegate = new RecordingWeightLoader();
        TensorParallelWeightLoader loader = loader(delegate, new StaticTensorParallelContext(1, 4));

        loader.load("model.layers.0.mlp.gate_proj.weight").close();
        assertEquals(new TensorShardSpec(TensorShardAxis.ROWS, 2304, 4608), delegate.lastShardSpec);

        loader.load("model.layers.0.mlp.up_proj.weight").close();
        assertEquals(new TensorShardSpec(TensorShardAxis.ROWS, 2304, 4608), delegate.lastShardSpec);

        loader.load("model.layers.0.mlp.down_proj.weight").close();
        assertEquals(new TensorShardSpec(TensorShardAxis.COLUMNS, 2304, 4608), delegate.lastShardSpec);
    }

    @Test
    public void replicatedAndDisabledTensorParallelLoadFullWeights() {
        RecordingWeightLoader delegate = new RecordingWeightLoader();
        TensorParallelWeightLoader enabled = loader(delegate, new StaticTensorParallelContext(1, 4));

        enabled.load("model.layers.0.input_layernorm.weight").close();
        assertEquals("model.layers.0.input_layernorm.weight", delegate.lastFullLoad);
        assertNull(delegate.lastShardSpec);

        TensorParallelWeightLoader disabled = loader(delegate, new StaticTensorParallelContext(0, 1));
        disabled.load("model.layers.0.self_attn.q_proj.weight").close();
        assertEquals("model.layers.0.self_attn.q_proj.weight", delegate.lastFullLoad);
        assertNull(delegate.lastShardSpec);
    }

    private static TensorParallelWeightLoader loader(RecordingWeightLoader delegate, TensorParallelContext context) {
        TensorParallelShardPlan plan = new TensorParallelShardPlan(
                TensorParallelPlanner.range(8, context),
                TensorParallelPlanner.range(4, context),
                TensorParallelPlanner.range(2048, context),
                TensorParallelPlanner.range(1024, context),
                TensorParallelPlanner.range(9216, context)
        );
        return new TensorParallelWeightLoader(delegate, context, plan, new DefaultTransformerWeightPolicyResolver());
    }

    private static final class RecordingWeightLoader implements WeightLoader {
        private String lastFullLoad;
        private TensorShardSpec lastShardSpec;

        @Override
        public Map<String, String> metadata() {
            return Map.of();
        }

        @Override
        public Map<String, TensorInfo> tensorInfoMap() {
            return Map.of();
        }

        @Override
        public AbstractTensor load(String name) {
            lastFullLoad = name;
            lastShardSpec = null;
            return new FloatBufferTensor(1, 1);
        }

        @Override
        public AbstractTensor load(String name, TensorShardSpec shardSpec) {
            lastFullLoad = null;
            lastShardSpec = shardSpec;
            return new FloatBufferTensor(1, 1);
        }

        @Override
        public DType getModelDType() {
            return DType.F32;
        }

        @Override
        public void close() {
        }
    }
}
