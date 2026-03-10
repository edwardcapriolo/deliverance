package io.teknek.deliverance.model;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.model.gemma2.GemmaTokenizer;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorCache;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class ChoiceEncodedTest {

    @Test
    void buildEncoded(){
        ModelFetcher fetch = new ModelFetcher("tjake", "gemma-2-2b-it-JQ4");
        File f = fetch.maybeDownload();
        MetricRegistry mr = new MetricRegistry();
        TensorCache tensorCache = new TensorCache(mr);
        try (AbstractModel m = ModelSupport.loadModel(f, DType.F32, DType.I8, new ConfigurableTensorProvider(tensorCache),
                mr, tensorCache, new KvBufferCacheSettings(true), fetch)) {
            ChoiceEncoded ci = new ChoiceEncoded(Arrays.asList("Giants", "Jets"), m.tokenizer);
            assertEquals( GemmaTokenizer.class, m.tokenizer.getClass());
            GemmaTokenizer gemmaT = (GemmaTokenizer) m.tokenizer;
            //Important: just because a model has the complete token in the vocabulary it might not be an optioa after
            //forward pass
            assertEquals(2, ci.getEncoded().size());
            assertEquals(List.of(218954L), ci.getEncoded().get("Giants"));
            assertEquals("Giants", m.tokenizer.decode(218954));
        }
    }

    @Test
    void longerName(){
        ModelFetcher fetch = new ModelFetcher("tjake", "gemma-2-2b-it-JQ4");
        File f = fetch.maybeDownload();
        MetricRegistry mr = new MetricRegistry();
        TensorCache tensorCache = new TensorCache(mr);
        try (AbstractModel m = ModelSupport.loadModel(f, DType.F32, DType.I8, new ConfigurableTensorProvider(tensorCache),
                mr, tensorCache, new KvBufferCacheSettings(true), fetch)) {
            ChoiceEncoded ci = new ChoiceEncoded(Arrays.asList("New York football Giants"), m.tokenizer);
            assertEquals( GemmaTokenizer.class, m.tokenizer.getClass());
            GemmaTokenizer gemmaT = (GemmaTokenizer) m.tokenizer;
            assertEquals(1, ci.getEncoded().size());
            assertEquals(Arrays.asList(2441L, 249L, 46671L, 249L, 43563L, 249L, 218954L), ci.getEncoded().get("New York football Giants"));

        }
    }
}
