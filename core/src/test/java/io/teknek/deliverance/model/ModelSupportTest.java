package io.teknek.deliverance.model;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.tensor.*;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ModelSupportTest {

    String modelName = "Qwen3-0.6B-JQ4";
    String modelOwner = "edwardcapriolo";
    ModelFetcher fetch = new ModelFetcher(modelOwner, modelName);

    @Test
    void load() {

        ArrayQueueTensorAllocator tc = new ArrayQueueTensorAllocator(new com.codahale.metrics.MetricRegistry());
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores());
             AbstractModel abstractModel = AutoModelForCausaLm.newBuilder(fetch)
                     .withWorkingMemoryType(DType.F32)
                     .withWorkingQuantType(DType.F32)
                     .withTensorAllocator(tc)
                     .withWrappedForkJoinPool(pool)
                     .withTensorProvider(new ConfigurableTensorProvider(tc, pool))
                     .buildLocalTransformerModel()) {

            assertEquals(io.teknek.deliverance.grace.Quen2Tokenizer.class, abstractModel.getTokenizer().getClass());
            {
                String prompt = "What comes next in the sequence? 1, 2, 3 ";
                PromptContext ctx = PromptContext.of(prompt);
                UUID u = UUID.randomUUID();
                Response r = abstractModel.generate(u, ctx, new GeneratorParameters().withSeed(43)
                        .withNtokens(50), new DoNothingGenerateEvent());
                assertEquals("4, 5, 6, 7, 8, 9, 10, 11, 12, 13,", r.responseText);
            }
            //Do it again
            {
                String prompt = "What comes next in the sequence? 1, 2 ";
                PromptContext ctx = PromptContext.of(prompt);
                UUID u = UUID.randomUUID();
                Response r = abstractModel.generate(u, ctx, new GeneratorParameters().withSeed(43)
                        .withNtokens(50), new DoNothingGenerateEvent());
                assertEquals("1/2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,", r.responseText);
            }

            {
                String prompt = "What comes next in the sequence? 1, 2, 3 ";
                PromptContext ctx = PromptContext.of(prompt);
                UUID u = UUID.randomUUID();
                Response r = abstractModel.generate(u, ctx, new GeneratorParameters().withSeed(43)
                        .withNtokens(50), new DoNothingGenerateEvent());
                assertEquals("4, 5, 6, 7, 8, 9, 10, 11, 12, 13,", r.responseText);
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
                //1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3
                //assertEquals("3, are the key technologies, technologies, technologies, technologies, technologies, technologies, technologies, technologies, technologies, technologies", r.responseText);
                assertTrue(r.responseText.contains("3"));
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
