package io.teknek.deliverance.model;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.tensor.*;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.toolcallparser.DefaultToolCallParser;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentMap;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class ModelSupportTest {

    String modelName = "microlama-lidor-finetuned";
    String modelOwner = "lidoreliya13";
    ModelFetcher fetch = new ModelFetcher(modelOwner, modelName);

    @Test
    void load() {

        File f = fetch.maybeDownload();
        MetricRegistry mr = new MetricRegistry();
        ArrayQueueTensorAllocator tc = new ArrayQueueTensorAllocator(new MetricRegistry());
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores());

             AbstractModel abstractModel = ModelSupport.loadModel(f, DType.F32, DType.F32,
                     new ConfigurableTensorProvider(tc, pool), mr, new ArrayQueueTensorAllocator(mr),
                     new KvBufferCacheSettings(true), fetch, new DefaultToolCallParser(), pool)) {

            assertEquals(io.teknek.deliverance.grace.GemmaTokenizer.class, abstractModel.getTokenizer().getClass());
            {
                String prompt = "What comes next in the sequence? 1, 2, 3 ";
                PromptContext ctx = PromptContext.of(prompt);
                UUID u = UUID.randomUUID();
                Response r = abstractModel.generate(u, ctx, new GeneratorParameters().withSeed(43)
                        .withNtokens(50), new DoNothingGenerateEvent());
                assertEquals("1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3", r.responseText);
            }
            //Do it again
            {
                String prompt = "What comes next in the sequence? 1, 2 ";
                PromptContext ctx = PromptContext.of(prompt);
                UUID u = UUID.randomUUID();
                Response r = abstractModel.generate(u, ctx, new GeneratorParameters().withSeed(43)
                        .withNtokens(50), new DoNothingGenerateEvent());
                assertEquals("3 are the next 1, 2, 3 are the next 2, and 4 are the next 3. 1", r.responseText);
            }

            {
                String prompt = "What comes next in the sequence? 1, 2, 3 ";
                PromptContext ctx = PromptContext.of(prompt);
                UUID u = UUID.randomUUID();
                Response r = abstractModel.generate(u, ctx, new GeneratorParameters().withSeed(43)
                        .withNtokens(50), new DoNothingGenerateEvent());
                assertEquals("1, 2, 3, 4, 5, 6, ", r.responseText);
            }
        }
    }

    @Test
    public void diskBasedKv() throws IOException {
        File f = new File("target/test-data");
        f.mkdir();
        KvBufferCacheSettings k = new KvBufferCacheSettings(f);
        try (AbstractModel model = AutoModelForCausaLm.newBuilder(this.fetch).withKvBufferCacheSettings(k).buildLocalTransformerModel() ){
            {
                String prompt = "What comes next in the sequence? 1, 2, 3 ";
                PromptContext ctx = PromptContext.of(prompt);
                Response r = model.generate(UUID.randomUUID(), ctx, new GeneratorParameters().withSeed(43)
                        .withNtokens(50), new DoNothingGenerateEvent());
                assertEquals("3, are the key technologies, technologies, technologies, technologies, technologies, technologies, technologies, technologies, technologies, technologies", r.responseText);
            }
        }
        Path directory = Paths.get(f.toURI());
        Files.walkFileTree(directory, new SimpleFileVisitor<Path>() {
            @Override
            public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
                Files.delete(file);
                return FileVisitResult.CONTINUE;
            }

            @Override
            public FileVisitResult postVisitDirectory(Path dir, IOException exc) throws IOException {
                Files.delete(dir);
                return FileVisitResult.CONTINUE;
            }
        });
    }
}
