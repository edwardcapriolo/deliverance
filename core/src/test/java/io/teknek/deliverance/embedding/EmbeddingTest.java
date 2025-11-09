package io.teknek.deliverance.embedding;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.ModelSupport;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorCache;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class EmbeddingTest {

    String text = "This is a test document about machine learning";
    @Test
    void embeddingAvg(){
        ModelFetcher fetch = new ModelFetcher("sentence-transformers", "all-MiniLM-L6-v2");
        File f = fetch.maybeDownload();
        MetricRegistry mr = new MetricRegistry();
        TensorCache tensorCache = new TensorCache(mr);
        try (AbstractModel model = ModelSupport.loadEmbeddingModel(f, DType.F32, DType.F32, new ConfigurableTensorProvider(tensorCache),
                new MetricRegistry(), tensorCache, new KvBufferCacheSettings(true))) {
            long [] ids = model.getTokenizer().encode(text);
            assertEquals("[101, 2023, 2003, 1037, 3231, 6254, 2055, 3698, 4083, 102]", Arrays.toString(ids));
            float[] embedding = model.embed(text, PoolingType.AVG);
            //  [0] = -9.4317534E-4
            //  [1] = 0.0065326607
            assertEquals(-9.4317534E-4, embedding[0], 0.0000001);
            assertEquals(0.0065326607, embedding[1], 0.0000001);
        }
    }

    @Test
    void embeddingModel(){
        ModelFetcher fetch = new ModelFetcher("sentence-transformers", "all-MiniLM-L6-v2");
        File f = fetch.maybeDownload();
        MetricRegistry mr = new MetricRegistry();
        TensorCache tensorCache = new TensorCache(mr);
        try (AbstractModel model = ModelSupport.loadEmbeddingModel(f, DType.F32, DType.F32, new ConfigurableTensorProvider(tensorCache),
                new MetricRegistry(), tensorCache, new KvBufferCacheSettings(true))) {
            long [] ids = model.getTokenizer().encode(text);
            assertEquals("[101, 2023, 2003, 1037, 3231, 6254, 2055, 3698, 4083, 102]", Arrays.toString(ids));
            float[] embedding = model.embed(text, PoolingType.MODEL);
            assertEquals(0.043238960206508636, embedding[0], 0.0000001);
            assertEquals(-0.051459357142448425, embedding[1], 0.0000001);
        }
    }
}
