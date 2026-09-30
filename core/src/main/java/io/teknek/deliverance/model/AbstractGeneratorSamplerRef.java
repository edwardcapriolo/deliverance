package io.teknek.deliverance.model;

import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator2.LayerNorm2;
import io.teknek.deliverance.guided.LogitsProcessorRef;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor2.ArgMax;
import io.teknek.deliverance.tensor2.BatchDotProduct;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.Scale;
import io.teknek.deliverance.tensor2.TensorRef;

import java.util.Optional;
import java.util.Random;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ForkJoinTask;

/** Base for sampling directly from TensorRef forward results. */
abstract class AbstractGeneratorSamplerRef {
    protected final AbstractModel model;
    protected final GeneratorParameters parameters;
    protected final TensorRef output;
    protected final TensorRef logits;
    protected final TensorRef argMaxScratch;
    protected final LayerNorm2 layerNorm;
    protected final Lighter lighter;
    protected final Random random;
    protected final ResponseContext responseContext;
    protected final Optional<LogitsProcessorRef> logitsProcessor;

    AbstractGeneratorSamplerRef(AbstractModel model, GeneratorParameters parameters, TensorRef output,
            TensorRef logits, LayerNorm2 layerNorm, Random random, ResponseContext responseContext,
            TensorRef argMaxScratch, Optional<LogitsProcessorRef> logitsProcessor) {
        this.model = model;
        this.parameters = parameters;
        this.output = output;
        this.logits = logits;
        this.layerNorm = layerNorm;
        this.random = random;
        this.responseContext = responseContext;
        this.argMaxScratch = argMaxScratch;
        this.logitsProcessor = logitsProcessor;
        this.lighter = model.getLighter();
    }

    public abstract SamplerReturn sample();

    protected void outputProjection(TensorRef embedding) {
        lighter.clear(logits);
        int vocabularySize = model.getConfig().vocabularySize;
        int workerCount = Math.max(1, model.getPool().getCoreCount());
        int chunkSize = Math.max(1, (vocabularySize + workerCount - 1) / workerCount);
        List<ForkJoinTask<?>> tasks = new ArrayList<>();
        for (int start = 0; start < vocabularySize; start += chunkSize) {
            int chunkStart = start;
            int count = Math.min(chunkSize, vocabularySize - chunkStart);
            tasks.add(model.getPool().getUnderlying().submit(() -> lighter.dotProductRows(logits, embedding,
                    model.sampleOutputRef.outputLogitsWeights(), 0, model.getConfig().embeddingLength,
                    chunkStart, count, chunkStart)));
        }
        tasks.forEach(ForkJoinTask::join);
    }

    protected void scale(float factor) {
        lighter.scale(new Scale(factor).target(logits).offsetAndLength(0, model.getConfig().vocabularySize));
    }

    protected int argMax() {
        lighter.argMax(new ArgMax(logits).into(argMaxScratch).offsetAndLength(0, model.getConfig().vocabularySize));
        return (int) argMaxScratch.get(0, 0);
    }
}
