package io.teknek.deliverance.model;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.tensor.AbstractTensor;
import org.junit.jupiter.api.Test;

import java.util.Comparator;
import java.util.List;
import java.util.UUID;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class Gemma2OutputHeadQuantizationParityIT {

    @Test
    public void q4OutputHeadPreservesFirstTokenAndTopLogitsForFixedPrompts() {
        ModelFetcher fetcher = new ModelFetcher("tjake", "gemma-2-2b-it-JQ4");
        try (AbstractModel denseOutput = AutoModelForCausaLm.newBuilder(fetcher).buildLocalTransformerModel();
             AbstractModel q4Output = AutoModelForCausaLm.newBuilder(fetcher)
                     .withOutputHeadQuantization(DType.Q4)
                     .buildLocalTransformerModel()) {
            for (String prompt : List.of(
                    "What is 1 + 1?",
                    "Explain tensor parallelism in one sentence.",
                    "Write a Python function that returns the larger of two integers.")) {
                LogitReport report = compareFirstTokenLogits(denseOutput, q4Output, prompt);
                System.out.println(report);
                assertEquals(report.denseArgmax(), report.q4Argmax(), "argmax changed for prompt: " + prompt);
                assertTrue(report.top5Overlap() >= 35, "top-n overlap too low for prompt: " + prompt);

                int denseGenerated = denseOutput.generate(UUID.randomUUID(), PromptContext.of(prompt),
                        new GeneratorParameters().withTemperature(0.0f).withMaxTokens(1).withSeed(123),
                        new DoNothingGenerateEvent()).generatedTokens.getFirst();
                int q4Generated = q4Output.generate(UUID.randomUUID(), PromptContext.of(prompt),
                        new GeneratorParameters().withTemperature(0.0f).withMaxTokens(1).withSeed(123),
                        new DoNothingGenerateEvent()).generatedTokens.getFirst();
                assertEquals(denseGenerated, q4Generated, "generated first token changed for prompt: " + prompt);
            }
        }
    }

    private static LogitReport compareFirstTokenLogits(AbstractModel denseOutput, AbstractModel q4Output, String prompt) {
        try (AbstractTensor denseLast = denseOutput.batchForward(denseOutput.constructPromptTokensForRuntime(prompt), 0);
             AbstractTensor q4Last = q4Output.batchForward(q4Output.constructPromptTokensForRuntime(prompt), 0);
             AbstractTensor denseLogits = logits(denseOutput, denseLast);
             AbstractTensor q4Logits = logits(q4Output, q4Last)) {
            int denseArgmax = argmax(denseLogits);
            int q4Argmax = argmax(q4Logits);
            List<Integer> denseTop5 = topK(denseLogits, 40);
            List<Integer> q4Top5 = topK(q4Logits, 40);
            long overlap = denseTop5.stream().filter(q4Top5::contains).count();
            double sumAbs = 0.0;
            double maxAbs = 0.0;
            for (int i = 0; i < denseLogits.shape().last(); i++) {
                double diff = Math.abs(denseLogits.get(0, i) - q4Logits.get(0, i));
                sumAbs += diff;
                maxAbs = Math.max(maxAbs, diff);
            }
            return new LogitReport(prompt, denseArgmax, q4Argmax, denseTop5, q4Top5, (int) overlap,
                    maxAbs, sumAbs / denseLogits.shape().last());
        }
    }

    private static AbstractTensor logits(AbstractModel model, AbstractTensor last) {
        AbstractTensor logits = model.makeDenseTensor(model.getConfig().vocabularySize);
        try (AbstractTensor embedding = model.sampleOutput.getOutputLayerNorm().forward(last.slice(last.shape().first() - 1))) {
            model.configurableTensorProvider.get().batchDotProduct(logits, embedding,
                    model.sampleOutput.getOutputLogitsWeights(), 0, 0, model.getConfig().embeddingLength);
        }
        return logits;
    }

    private static int argmax(AbstractTensor logits) {
        int best = 0;
        float bestValue = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < logits.shape().last(); i++) {
            float value = logits.get(0, i);
            if (value > bestValue) {
                best = i;
                bestValue = value;
            }
        }
        return best;
    }

    private static List<Integer> topK(AbstractTensor logits, int k) {
        return IntStream.range(0, logits.shape().last())
                .boxed()
                .sorted(Comparator.comparingDouble((Integer i) -> logits.get(0, i)).reversed())
                .limit(k)
                .toList();
    }

    private record LogitReport(String prompt, int denseArgmax, int q4Argmax, List<Integer> denseTop5,
            List<Integer> q4Top5, int top5Overlap, double maxAbsDiff, double meanAbsDiff) {
    }
}
