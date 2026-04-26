package io.teknek.deliverance.tensor;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.safetensors.DistributedContext;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class KvBufferCachePrefixTest {

    private AbstractModel mockModel(){
        Config config = new Config(128, 64,128,4,
                4,2,1e-5f,1000,0,
                List.of(1), ActivationFunction.Type.GELU_PYTORCH_TANH, null,null);
        DistributedContext dctx = DistributedContext.builder(config).build();
        config.setDistributedContext(dctx);
        AbstractModel model = mock (AbstractModel.class);
        when(model.getConfig()).thenReturn(config);
        when(model.getWorkingDType()).thenReturn(DType.F32);
                when(model.getTensorAllocator()).thenReturn( new ArrayQueueTensorAllocator(new MetricRegistry()));
                when(model.getMetricRegistry()).thenReturn(new MetricRegistry());
        return model;
    }

    @Test
    public void testExactPrefixHit(){
        KvBufferCacheSettings settings = new KvBufferCacheSettings(true).withMaxEntries(512);
        KvBufferCache cache = new KvBufferCache(mockModel(), settings);
        int [] tokens = { 1, 2, 3, 4, 5 ,6,7, 8 ,9  };
        KvBufferCache.KvBuffer buf = cache.getEphemeralKvBuffer();
        cache.storePrefix(tokens, buf);

        {
            int [] tokens2 = { 1, 2, 3 };
            KvBufferCache.KvBuffer buf2 = cache.getEphemeralKvBuffer();
            cache.storePrefix(tokens2, buf2);
        }

        KvBufferCache.PrefixEntry e = cache.lookupPrefix(tokens);
        assertEquals(tokens.length - 1, e.length);

    }
}
