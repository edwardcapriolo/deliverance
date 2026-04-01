package net.deliverance.distributed;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.tensor.TensorCache;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class DeliveranceServiceTest {

    @Test
    public void seeSplits(){
        try ( WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores())) {
            AbstractModel m = AutoModelForCausaLm.newBuilder(new ModelFetcher("tjake", "TinyLlama-1.1B-Chat-v1.0-Jlama-Q4"))
                    .withTensorProvider(new ConfigurableTensorProvider(new TensorCache(new MetricRegistry()), pool)).build();
            DeliveranceService d = new DeliveranceService(m, 10, true, true);
            assertEquals(4, d.getHeadsPerLayerShard());
            assertEquals(2, d.getLayersPerShard());
            assertEquals(1, d.getNumHeadShards());
            assertEquals(11, d.getNumLayerShards());
            assertEquals(11, d.getOrdinalCombinations().size());
            assertEquals("[[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0], [10, 0]]", d.getOrdinalCombinations().toString());

        }
    }
}
