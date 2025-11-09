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

public class EmbeddingTest {


    @Test
    void embeddingGo(){
        //"sentence-transformers/all-MiniLM-L6-v2"
        ModelFetcher fetch = new ModelFetcher("sentence-transformers", "all-MiniLM-L6-v2");
        File f = fetch.maybeDownload();
        MetricRegistry mr = new MetricRegistry();
        TensorCache tensorCache = new TensorCache(mr);
        try (AbstractModel model = ModelSupport.loadEmbeddingModel(f, DType.F32, DType.F32, new ConfigurableTensorProvider(tensorCache),
                new MetricRegistry(), tensorCache, new KvBufferCacheSettings(true))) {
            String text = "This is a test document about machine learning";
            long [] ids = model.getTokenizer().encode(text);
            Assertions.assertEquals("[101, 2023, 2003, 1037, 3231, 6254, 2055, 3698, 4083, 102]", Arrays.toString(ids));
            float[] embedding = model.embed(text, PoolingType.AVG);

            System.out.println("First 10 values:");
            for (int i = 0; i < 10; i++) {
                System.out.println("  [" + i + "] = " + embedding[i]);
            }
        }
    }
}
