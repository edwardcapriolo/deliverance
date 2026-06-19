package io.teknek.deliverance.model.hf;

import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.KvBufferCache;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** Ports feasible text-model checks from HF {@code ModelTesterMixin} and {@code CausalLMModelTest}. */
public interface HfModelTesterMixinPort extends HfTextCommonTestAdapter {
    @Test
    default void hfCausalLmModelTestModelEquivalent() {
        Path modelDir = writeTinyCheckpoint("hf-causal-lm-model", 21_001);
        AbstractModel model = loadTinyModel(modelDir);
        try (KvBufferCache.KvBuffer kv = model.newKvBuffer();
             AbstractTensor output = model.batchForward(hfSampleTokenIds(), 0, kv)) {
            assertEquals(hfSampleTokenIds().length, output.shape().first());
            assertEquals(model.getConfig().embeddingLength, output.shape().last());
            assertFiniteTensor(output);
        } finally {
            closeModel(model);
        }
    }

    @Test
    default void hfModelTesterSaveLoadForwardParityEquivalent() {
        Path modelDir = writeTinyCheckpoint("hf-model-save-load", 21_002);
        int[] tokens = hfSampleTokenIds();
        AbstractModel firstModel = loadTinyModel(modelDir);
        AbstractModel secondModel = loadTinyModel(modelDir);
        try (KvBufferCache.KvBuffer firstKv = firstModel.newKvBuffer();
             KvBufferCache.KvBuffer secondKv = secondModel.newKvBuffer();
             AbstractTensor first = firstModel.batchForward(tokens, 0, firstKv);
             AbstractTensor second = secondModel.batchForward(tokens, 0, secondKv)) {
            assertTensorsClose(first, second, 0.0f, "same checkpoint loaded twice should produce identical output");
        } finally {
            closeModel(firstModel);
            closeModel(secondModel);
        }
    }

    @Test
    default void hfModelTesterDeterminismEquivalent() {
        Path modelDir = writeTinyCheckpoint("hf-model-determinism", 21_003);
        int[] tokens = hfSampleTokenIds();
        AbstractModel model = loadTinyModel(modelDir);
        try (KvBufferCache.KvBuffer firstKv = model.newKvBuffer();
             KvBufferCache.KvBuffer secondKv = model.newKvBuffer();
             AbstractTensor first = model.batchForward(tokens, 0, firstKv);
             AbstractTensor second = model.batchForward(tokens, 0, secondKv)) {
            assertTensorsClose(first, second, 0.0f, "same model/input should be deterministic");
        } finally {
            closeModel(model);
        }
    }

    @Test
    default void hfModelTesterDifferentWeightsChangeForwardOutputEquivalent() {
        Path firstModelDir = writeTinyCheckpoint("hf-model-different-weights-first", 21_005);
        Path secondModelDir = writeTinyCheckpoint("hf-model-different-weights-second", 21_105);
        int[] tokens = hfSampleTokenIds();
        AbstractModel firstModel = loadTinyModel(firstModelDir);
        AbstractModel secondModel = loadTinyModel(secondModelDir);
        try (KvBufferCache.KvBuffer firstKv = firstModel.newKvBuffer();
             KvBufferCache.KvBuffer secondKv = secondModel.newKvBuffer();
             AbstractTensor first = firstModel.batchForward(tokens, 0, firstKv);
             AbstractTensor second = secondModel.batchForward(tokens, 0, secondKv)) {
            assertOutputsDiffer(first, second, "different checkpoints should not collapse to the same output");
        } finally {
            closeModel(firstModel);
            closeModel(secondModel);
        }
    }

    @Test
    default void hfModelTesterForwardFiniteAcrossSequenceLengthsEquivalent() {
        Path modelDir = writeTinyCheckpoint("hf-model-forward-lengths", 21_006);
        int[] tokens = hfSampleTokenIds();
        AbstractModel model = loadTinyModel(modelDir);
        try {
            for (int length = 1; length <= tokens.length; length++) {
                int[] prefix = java.util.Arrays.copyOf(tokens, length);
                try (KvBufferCache.KvBuffer kv = model.newKvBuffer();
                     AbstractTensor output = model.batchForward(prefix, 0, kv)) {
                    assertEquals(length, output.shape().first(), "forward row count for length=" + length);
                    assertEquals(model.getConfig().embeddingLength, output.shape().last(),
                            "forward width for length=" + length);
                    assertFiniteTensor(output);
                }
            }
        } finally {
            closeModel(model);
        }
    }

    @Test
    default void hfModelTesterPastKeyValuesFormatEquivalent() {
        Path modelDir = writeTinyCheckpoint("hf-model-past-kv-format", 21_004);
        AbstractModel model = loadTinyModel(modelDir);
        try (KvBufferCache.KvBuffer kv = model.newKvBuffer();
             AbstractTensor output = model.batchForward(hfSampleTokenIds(), 0, kv);
             AbstractTensor key = kv.getKeyTensorForPosition(0, 0);
             AbstractTensor value = kv.getValTensorForPosition(0, 0)) {
            assertFiniteTensor(output);
            assertEquals(model.getLocalKvLength(), key.shape().last());
            assertEquals(model.getLocalKvLength(), value.shape().last());
            assertFiniteTensor(key);
            assertFiniteTensor(value);
        } finally {
            closeModel(model);
        }
    }

    private void assertOutputsDiffer(AbstractTensor first, AbstractTensor second, String message) {
        assertEquals(first.shape().first(), second.shape().first(), message + " rows");
        assertEquals(first.shape().last(), second.shape().last(), message + " cols");
        float maxAbs = 0.0f;
        for (int row = 0; row < first.shape().first(); row++) {
            for (int col = 0; col < first.shape().last(); col++) {
                maxAbs = Math.max(maxAbs, Math.abs(first.get(row, col) - second.get(row, col)));
            }
        }
        org.junit.jupiter.api.Assertions.assertTrue(maxAbs > 1.0e-6f, message + " maxAbs=" + maxAbs);
    }
}
