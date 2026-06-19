package io.teknek.deliverance.model.hf;

import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.KvBufferCache;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertThrows;

/** Ports feasible text-only checks from HF {@code GenerationTesterMixin}. */
public interface HfGenerationTesterMixinPort extends HfTextCommonTestAdapter {
    @Test
    default void hfGenerationContinueFromPastKeyValuesEquivalent() {
        Path modelDir = writeTinyCheckpoint("hf-generation-continue-past-kv", 22_001);
        int[] prompt = hfSampleTokenIds();
        int[] continuation = new int[]{prompt[prompt.length - 1] + 1, prompt[prompt.length - 1] + 2};
        AbstractModel model = loadTinyModel(modelDir);
        try (KvBufferCache.KvBuffer decodeKv = model.newKvBuffer();
             AbstractTensor promptOutput = model.batchForward(prompt, 0, decodeKv)) {
            assertFiniteTensor(promptOutput);
            for (int i = 0; i < continuation.length; i++) {
                try (AbstractTensor decode = model.forward(continuation[i], prompt.length + i, decodeKv);
                     AbstractTensor replay = model.batchForward(tokensWithContinuation(prompt, continuation, i + 1), 0);
                     AbstractTensor replayLastRow = lastRow(replay)) {
                    assertTensorsClose(replayLastRow, decode, 1.0e-4f,
                            "decode from cached KV should match cold replay at continuation step " + i);
                }
            }
        } finally {
            closeModel(model);
        }
    }

    @Test
    default void hfGenerationInputsEmbedsUnsupportedWhenModelRequiresInputIds() {
        Path modelDir = writeTinyCheckpoint("hf-generation-inputs-embeds-reject", 22_002);
        AbstractModel model = loadTinyModel(modelDir);
        try (KvBufferCache.KvBuffer kv = model.newKvBuffer();
             AbstractTensor inputsEmbeds = makeInputsEmbeds(1, model.getConfig().embeddingLength, 22_003)) {
            assertThrows(UnsupportedOperationException.class,
                    () -> model.forward(inputsEmbeds, 0, kv, Optional.empty()));
        } finally {
            closeModel(model);
        }
    }

    private static int[] tokensWithContinuation(int[] prompt, int[] continuation, int continuationLength) {
        int[] tokens = Arrays.copyOf(prompt, prompt.length + continuationLength);
        System.arraycopy(continuation, 0, tokens, prompt.length, continuationLength);
        return tokens;
    }

    private static AbstractTensor lastRow(AbstractTensor tensor) {
        return tensor.slice(true, tensor.shape().first() - 1);
    }
}
