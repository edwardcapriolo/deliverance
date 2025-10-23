package io.teknek.deliverance.tokenizer;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.fetch.ModelFetcher;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.ModelSupport;
import io.teknek.deliverance.model.llama.LlamaTokenizer;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorCache;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class LlamaTokenizerTest {

    @Test
    void encodingDecodingTest(){
        String modelName = "TinyLlama-1.1B-Chat-v1.0-Jlama-Q4";
        String modelOwner = "tjake";
        ModelFetcher fetch = new ModelFetcher(modelOwner, modelName);
        File f = fetch.maybeDownload();
        TensorCache tc = new TensorCache(new MetricRegistry());
        try (AbstractModel m = ModelSupport.loadModel(f, DType.F32, DType.I8, new ConfigurableTensorProvider(tc),
                new MetricRegistry(), tc, new KvBufferCacheSettings(true))) {
            List<String> tokens = m.getTokenizer().tokenize("show me the money!");
            assertEquals(List.of("show me the money!"), tokens);
            long[] encode = m.getTokenizer().encode("show me!");
            assertArrayEquals(new long[]{4294, 35, 1004, 29991}, encode);
            assertEquals("show", m.getTokenizer().decode(4294));
            assertEquals("me", m.getTokenizer().decode(1004));
            assertEquals("!", m.getTokenizer().decode(29991));
        }
    }

    @Test
    void merges(){
        String modelName = "TinyLlama-1.1B-Chat-v1.0-Jlama-Q4";
        String modelOwner = "tjake";
        ModelFetcher fetch = new ModelFetcher(modelOwner, modelName);
        File f = fetch.maybeDownload();
        TensorCache tc = new TensorCache(new MetricRegistry())  ;
        try (AbstractModel m = ModelSupport.loadModel(f, DType.F32, DType.I8, new ConfigurableTensorProvider(tc),
                new MetricRegistry(), tc, new KvBufferCacheSettings(true))) {
            if (m.getTokenizer() instanceof LlamaTokenizer t){
                System.out.println(t.getModel().merges.size());

                //og e=44912, ▁acc omp=22400, ▁re move=6810, ▁disco very=43704, ▁e po=45284, ▁Intern et=10005, ▁erst mals=42498, ▁r aggi=44997, ax is=19626
                long[] encode = m.getTokenizer().encode(" ▁disco very");

                //assertArrayEquals(new long [] {43704}, encode);
            }
        }
    }

    @Disabled
    public void TestLLamaTokenizer() {
        //   String modelPrefix = "../models/Llama-2-7b-chat-hf-2";
        String modelName = "TinyLlama-1.1B-Chat-v1.0-Jlama-Q4";
        String modelOwner = "tjake";
        ModelFetcher fetch = new ModelFetcher(modelOwner, modelName);
        File f = fetch.maybeDownload();
               TensorCache tc = new TensorCache(new MetricRegistry())  ;
        try (AbstractModel m = ModelSupport.loadModel(f, DType.F32, DType.I8, new ConfigurableTensorProvider(tc),
                new MetricRegistry(), tc, new KvBufferCacheSettings(true))) {
            String p = "[INST] Tell me a joke. \uD83D\uDC31 [/INST] Answer ";
            if (m.getTokenizer() instanceof LlamaTokenizer tokenizer) {
                long[] actual = tokenizer.encode(p);
                System.out.println(Arrays.toString(actual));
                //                        [29961, 25580, 29962, 35, 29911, 514, 35, 1004, 35, 29874, 35, 2212, 446, 29889, 35, 243, 162, 147, 180, 35, 29961, 29914, 25580, 29962, 35, 22550, 35]
                long[] expected = new long[]{518, 25580, 29962, 24948, 592, 263, 2958, 446, 29889, 29871, 243, 162, 147, 180, 518, 29914, 25580,
                        29962, 673, 29871};

                assertArrayEquals(expected, actual);

                String out = tokenizer.decode(actual);
                assertEquals(p, out);

                String s = tokenizer.decode(518);
                assertEquals(" [", s);

                long[] token = tokenizer.encode(p + "\n");
                expected = new long[]{518, 25580, 29962, 24948, 592, 263, 2958, 446, 29889, 29871, 243, 162, 147, 180, 518, 29914, 25580, 29962,
                        673, 29871, 13};
                assertArrayEquals(expected, token);
            }
        }
    }

}
