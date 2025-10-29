package io.teknek.deliverance.model;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.llama.LlamaTokenizer;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.tensor.*;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentMap;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class ModelSupportTest {

    @Test
    void load(){
        String modelName = "microlama-lidor-finetuned";
        String modelOwner = "lidoreliya13";
        ModelFetcher fetch = new ModelFetcher(modelOwner, modelName);
        File f = fetch.maybeDownload();
        MetricRegistry mr = new MetricRegistry();
        TensorCache tc = new TensorCache(new MetricRegistry());
        try (AbstractModel abstractModel = ModelSupport.loadModel(f, DType.F32, DType.F32,
                new ConfigurableTensorProvider(tc), mr, new TensorCache(mr),
                new KvBufferCacheSettings(true))) {

            String prompt = "What comes next in the sequence? 1, 2, 3 ";
            PromptContext ctx = PromptContext.of(prompt);

            assertEquals(LlamaTokenizer.class, abstractModel.tokenizer.getClass());
            {
                UUID u = UUID.randomUUID();
                Response r = abstractModel.generate(u, ctx, new GeneratorParameters().withSeed(43).withNtokens(50), (s, f1) -> {
                });
                assertEquals("4,5,6,7,8,9,10,11,12,", r.responseText);
                ConcurrentMap<String, KvBufferCache.KvBuffer> d = abstractModel.kvBufferCache.getCacheByKey();
                assertEquals(d.size(), 1);
                Map.Entry<String, KvBufferCache.KvBuffer> entry = d.entrySet().iterator().next();
                assertEquals(entry.getKey(), "sha1obetter");
            }
            {
                UUID u = UUID.randomUUID();
                Response r = abstractModel.generate(u, ctx, new GeneratorParameters().withSeed(43).withNtokens(50), (s, f1) -> {
                });
                assertEquals("4", r.responseText);
                ConcurrentMap<String, KvBufferCache.KvBuffer> d = abstractModel.kvBufferCache.getCacheByKey();
                assertEquals(d.size(), 1);
                Map.Entry<String, KvBufferCache.KvBuffer> entry = d.entrySet().iterator().next();
                assertEquals(entry.getKey(), "sha1obetter");
            }
            {
                UUID u = UUID.randomUUID();
                Response r = abstractModel.generate(u, ctx, new GeneratorParameters().withSeed(43).withNtokens(50), (s, f1) -> {
                });
                assertEquals("4", r.responseText);
                ConcurrentMap<String, KvBufferCache.KvBuffer> d = abstractModel.kvBufferCache.getCacheByKey();
                assertEquals(d.size(), 1);
                Map.Entry<String, KvBufferCache.KvBuffer> entry = d.entrySet().iterator().next();
                assertEquals(entry.getKey(), "sha1obetter");
            }

            {
                UUID u = UUID.randomUUID();
                Response r = abstractModel.generate(u, ctx, new GeneratorParameters().withSeed(43).withNtokens(50).withSalt("yo"), (s, f1) -> {
                });
                assertEquals("4,5,6,7,8,9,10,11,12,", r.responseText);
                ConcurrentMap<String, KvBufferCache.KvBuffer> d = abstractModel.kvBufferCache.getCacheByKey();
                assertEquals(d.size(), 2);
                Map.Entry<String, KvBufferCache.KvBuffer> entry = d.entrySet().iterator().next();
            }
        }
    }

    @Test
    void dedicatedCacheTest(){
        String modelName = "microlama-lidor-finetuned";
        String modelOwner = "lidoreliya13";
        ModelFetcher fetch = new ModelFetcher(modelOwner, modelName);
        File f = fetch.maybeDownload();

        TensorCache tc = new TensorCache(new MetricRegistry());
        TensorCache dedicatedKvCache = new TensorCache(new MetricRegistry());
        try (AbstractModel abstractModel = ModelSupport.loadModel(f, DType.F32, DType.F32, new ConfigurableTensorProvider(tc),
                new MetricRegistry(), new TensorCache(new MetricRegistry()), new KvBufferCacheSettings(dedicatedKvCache))) {
            String prompt = "What comes next in the sequence? 1, 2, 3 ";
            PromptContext ctx = PromptContext.of(prompt);


            {
                UUID u = UUID.randomUUID();
                Response r = abstractModel.generate(u, ctx, new GeneratorParameters().withSeed(43).withNtokens(50), (s, f1) -> {
                });
                assertEquals("4,5,6,7,8,9,10,11,12,", r.responseText);
                ConcurrentMap<String, KvBufferCache.KvBuffer> d = abstractModel.kvBufferCache.getCacheByKey();
                assertEquals(d.size(), 1);
                Map.Entry<String, KvBufferCache.KvBuffer> entry = d.entrySet().iterator().next();
                assertEquals(entry.getKey(), "sha1obetter");
            }

        }

    }
}
