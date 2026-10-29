package io.teknek.deliverance.embedding;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForEmbeddings;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.util.Arrays;


import static org.junit.jupiter.api.Assertions.assertEquals;

public class EmbeddingTest {

    String text = "This is a test document about machine learning";

    @Test
    void embeddingAvg(){
        ModelFetcher fetch = new ModelFetcher("edwardcapriolo", "all-MiniLM-L6-v2-JQ4");
         AutoModelForEmbeddings.Builder builder = AutoModelForEmbeddings.newBuilder(fetch);
         builder.withWorkingMemoryType(DType.F32);
         builder.withWorkingQuantType(DType.F32);
         try (AbstractModel model = builder.buildLocalEmbeddingModel()) {
            int [] ids = model.getTokenizer().encode(text).inputIds();
            assertEquals("[101, 2023, 2003, 1037, 3231, 6254, 2055, 3698, 4083, 102]", Arrays.toString(ids));
            float[] embedding = model.embed(text, PoolingType.AVG);
            assertEquals(384, embedding.length, "Embedding should have 384 dimensions");
        }
    }

    @Test
    void embeddingModel(){
        ModelFetcher fetch = new ModelFetcher("edwardcapriolo", "all-MiniLM-L6-v2-JQ4");
         AutoModelForEmbeddings.Builder builder = AutoModelForEmbeddings.newBuilder(fetch);
         builder.withWorkingMemoryType(DType.F32);
         builder.withWorkingQuantType(DType.F32);
         try (AbstractModel model = builder.buildLocalEmbeddingModel()) {
                int[] ids = model.getTokenizer().encode(text).inputIds();
                assertEquals("[101, 2023, 2003, 1037, 3231, 6254, 2055, 3698, 4083, 102]", Arrays.toString(ids));
                model.embed(text, PoolingType.MODEL);
        }
    }
}
