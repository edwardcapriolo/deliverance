package io.teknek.deliverance.model;

import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.model.llama.LlamaHfTextModelPortedTest;
import io.teknek.deliverance.model.llama.LlamaModel;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.kv.KvCacheSession;
import io.teknek.deliverance.tensor2.TensorRef;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.Optional;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;

class LlamaTensorRefOutputCharacterizationTest {
    @TempDir
    Path tempDir;

    @Test
    void legacyAndTensorRefOutputNormLogitsAndGreedyTokenMatch() {
        Path modelDir = LlamaHfTextModelPortedTest.writeTinyCheckpoint(tempDir.resolve("llama-output-parity"),
                LlamaHfTextModelPortedTest.tinyConfig(), 209);
        int[] prompt = {3, 4, 5, 6};

        try (LlamaModel tensorRef = LlamaHfTextModelPortedTest.loadLlamaModel(modelDir, true);
             LlamaModel legacy = LlamaHfTextModelPortedTest.loadLlamaModel(modelDir, false);
             KvCacheSession refSession = tensorRef.newKvCacheSession();
             TensorRef refHidden = tensorRef.batchForwardRef(prompt, 0, refSession);
             TensorRef refLast = refHidden.slice((int) refHidden.shape().first() - 1);
             AbstractTensor legacyHidden = legacy.batchForward(prompt, 0);
             AbstractTensor legacyLast = legacyHidden.slice((int) legacyHidden.shape().first() - 1);
             AbstractTensor legacyLogits = legacy.makeDenseTensor(
                     TensorShape.of(1, legacy.getConfig().vocabularySize));
             AbstractTensor legacyScratch = legacy.makeDenseTensor(TensorShape.of(1, 2));
             TensorRef refLogits = tensorRef.makeDenseTensorRef(1, tensorRef.getConfig().vocabularySize);
             TensorRef refScratch = tensorRef.makeDenseTensorRef(1, 2)) {
            GeneratorParameters legacyParameters = new GeneratorParameters().withTemperature(0.0f);
            SamplerReturn legacySample = new DeliveranceSampler(legacy, legacyParameters, legacyLast, legacyLogits,
                    legacy.sampleOutput.getOutputLayerNorm(), new Random(1), 0.0f, new ResponseContext(legacy),
                    legacyScratch, Optional.empty()).sample();

            GeneratorParameters refParameters = new GeneratorParameters().withTemperature(0.0f);
            SamplerReturn refSample = new DeliveranceSamplerRef(tensorRef, refParameters, refLast, refLogits,
                    tensorRef.sampleOutputRef.outputLayerNorm(), tensorRef.sampleOutputRef.outputLogitsWeights(),
                    new Random(1), new ResponseContext(tensorRef), refScratch, Optional.empty()).sample();

            for (int token = 0; token < legacy.getConfig().vocabularySize; token++) {
                assertEquals(legacyLogits.get(0, token), refLogits.get(0, token), 1.0e-3f,
                        "logit token=" + token);
            }
            assertEquals(legacySample.token, refSample.token, "greedy token");
        }
    }
}
