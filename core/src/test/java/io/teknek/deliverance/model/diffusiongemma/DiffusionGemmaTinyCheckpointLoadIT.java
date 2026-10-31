package io.teknek.deliverance.model.diffusiongemma;

import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.safetensors.DefaultWeightLoader;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.kv.AttentionPattern;
import io.teknek.deliverance.tensor.kv.KvCacheSession;
import io.teknek.deliverance.tensor.kv.KvReadView;
import io.teknek.deliverance.tensor2.TensorRef;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

@Tag("longtest")
class DiffusionGemmaTinyCheckpointLoadIT {

    @Test
    void tinyDiffusionGemmaCheckpointCanOpenAndInstantiateSkeleton() {
        ModelFetcher fetcher = new ModelFetcher("trl-internal-testing", "tiny-DiffusionGemmaForBlockDiffusion");

        try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetcher).buildLocalTransformerModel()) {
            assertInstanceOf(DiffusionGemmaModel.class, model);
            DiffusionGemmaModel diffusionGemma = (DiffusionGemmaModel) model;
             try (TensorRef canvas = model.getLighter().allocate(io.teknek.deliverance.DType.F32,
                     TensorShape.of(1, 32));
                  DefaultWeightLoader weights = new DefaultWeightLoader(fetcher.pathForModel().toFile())) {
                for (int position = 0; position < canvas.shape().last(); position++) {
                    canvas.set(position, 0, position);
                }
                 try (TensorRef embeddings = diffusionGemma.embedCanvasTokensRef(canvas);
                      TensorRef selfConditioned = diffusionGemma.applySelfConditioningRef(embeddings, embeddings);
                      TensorRef embeddingWeights = weights.loadRef("model.decoder.embed_tokens.weight")) {
                    assertEquals(1, embeddings.shape().dim(0));
                    assertEquals(32, embeddings.shape().dim(1));
                    assertEquals(16, embeddings.shape().dim(2));
                    assertEquals(1, selfConditioned.shape().dim(0));
                    assertEquals(32, selfConditioned.shape().dim(1));
                    assertEquals(16, selfConditioned.shape().dim(2));
                    for (int position = 0; position < canvas.shape().last(); position++) {
                        for (int hidden = 0; hidden < 16; hidden++) {
                            assertEquals(embeddingWeights.get(position, hidden), embeddings.get(0, position, hidden),
                                    1.0e-6f, "position=" + position + " hidden=" + hidden);
                            assertTrue(Float.isFinite(selfConditioned.get(0, position, hidden)),
                                    "self-conditioned value must be finite position=" + position + " hidden=" + hidden);
                        }
                    }
                }
            }
        }
    }

    @Test
    void testModel() {
        ModelFetcher fetcher = new ModelFetcher("trl-internal-testing", "tiny-DiffusionGemmaForBlockDiffusion");

        try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetcher).buildLocalTransformerModel()) {
            DiffusionGemmaModel diffusionGemma = (DiffusionGemmaModel) model;
             try (TensorRef decoderInputIds = model.getLighter().allocate(io.teknek.deliverance.DType.F32,
                     TensorShape.of(3, 32))) {
                for (int batch = 0; batch < decoderInputIds.shape().first(); batch++) {
                    for (int position = 0; position < decoderInputIds.shape().last(); position++) {
                        decoderInputIds.set(position + 1, batch, position);
                    }
                }
                try (TensorRef output = diffusionGemma.forwardTextOnlyRef(decoderInputIds)) {
                    assertEquals(3, output.shape().dim(0));
                    assertEquals(32, output.shape().dim(1));
                    assertEquals(16, output.shape().dim(2));
                }
            }
        }
    }

    @Test
    void testModelLogitsShape() {
        ModelFetcher fetcher = new ModelFetcher("trl-internal-testing", "tiny-DiffusionGemmaForBlockDiffusion");

        try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetcher).buildLocalTransformerModel()) {
            DiffusionGemmaModel diffusionGemma = (DiffusionGemmaModel) model;
             try (TensorRef decoderInputIds = model.getLighter().allocate(io.teknek.deliverance.DType.F32,
                     TensorShape.of(1, 32))) {
                for (int position = 0; position < decoderInputIds.shape().last(); position++) {
                    decoderInputIds.set(position + 1, 0, position);
                }
                try (TensorRef output = diffusionGemma.forwardTextOnlyRef(decoderInputIds);
                     TensorRef logits = diffusionGemma.logitsForCanvasPositionRef(output, 0, 0)) {
                    assertEquals(1, logits.shape().dim(0));
                    assertEquals(262_144, logits.shape().dim(1));
                }
            }
        }
    }

    @Test
    void encoderTextCacheWritesKeyAndValueRows() {
        ModelFetcher fetcher = new ModelFetcher("trl-internal-testing", "tiny-DiffusionGemmaForBlockDiffusion");

         try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetcher).buildLocalTransformerModel();
              KvCacheSession kv = model.newKvCacheSession();
              TensorRef inputIds = model.getLighter().allocate(io.teknek.deliverance.DType.F32, TensorShape.of(1, 3))) {
            DiffusionGemmaModel diffusionGemma = (DiffusionGemmaModel) model;
            inputIds.set(1.0f, 0, 0);
            inputIds.set(2.0f, 0, 1);
            inputIds.set(3.0f, 0, 2);

             diffusionGemma.encodeTextToCacheRef(inputIds, kv);

             try (KvReadView view = kv.readView(0, 3, AttentionPattern.CAUSAL);
                  TensorRef key = view.keyRowCopyRef(0);
                  TensorRef value = view.valueRowCopyRef(0)) {
                assertEquals(16, key.shape().last());
                assertEquals(16, value.shape().last());
                assertTrue(nonZero(key));
                assertTrue(nonZero(value));
            }
        }
    }

    @Test
    void encoderTextLayerForwardProducesFiniteHiddenStates() {
        ModelFetcher fetcher = new ModelFetcher("trl-internal-testing", "tiny-DiffusionGemmaForBlockDiffusion");

         try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetcher).buildLocalTransformerModel();
              TensorRef inputIds = model.getLighter().allocate(io.teknek.deliverance.DType.F32,
                      TensorShape.of(2, 32))) {
            DiffusionGemmaModel diffusionGemma = (DiffusionGemmaModel) model;
            for (int batch = 0; batch < inputIds.shape().first(); batch++) {
                for (int position = 0; position < inputIds.shape().last(); position++) {
                    inputIds.set(position + batch + 1, batch, position);
                }
            }

             try (TensorRef output = diffusionGemma.forwardTextOnlyRef(inputIds)) {
                assertEquals(2, output.shape().dim(0));
                assertEquals(32, output.shape().dim(1));
                assertEquals(16, output.shape().dim(2));
                assertTrue(nonZero(output));
                assertFinite(output);
            }
        }
    }

    @Test
    void tinyCheckpointGenerateSmokeReturnsTokens() {
        ModelFetcher fetcher = new ModelFetcher("trl-internal-testing", "tiny-DiffusionGemmaForBlockDiffusion");

        try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetcher).buildLocalTransformerModel()) {
            Response response = model.generate(UUID.randomUUID(), PromptContext.of("hello"),
                    new GeneratorParameters().withMaxTokens(2), (next, nextRaw, nextCleaned, timing) -> { });

            assertEquals(2, response.generatedTokens.size());
            assertTrue(response.promptTokens > 0);
            assertTrue(response.responseText != null);
            assertTrue(response.responseTextWithSpecialTokens != null);
        }
    }

    private static boolean nonZero(TensorRef tensor) {
        return anyValue(tensor, value -> value != 0.0f);
    }

    private static void assertFinite(TensorRef tensor) {
        assertTrue(anyValue(tensor, Float::isFinite), "tensor must contain at least one finite value");
        forEachValue(tensor, value -> assertTrue(Float.isFinite(value), "tensor value must be finite"));
    }

    private static boolean anyValue(TensorRef tensor, FloatPredicate predicate) {
        boolean[] found = {false};
        forEachValue(tensor, value -> found[0] |= predicate.test(value));
        return found[0];
    }

    private static void forEachValue(TensorRef tensor, FloatConsumer consumer) {
        if (tensor.dims() == 2) {
            for (int row = 0; row < tensor.shape().first(); row++) {
                for (int col = 0; col < tensor.shape().last(); col++) {
                    consumer.accept(tensor.get(row, col));
                }
            }
            return;
        }
        if (tensor.dims() == 3) {
            for (int batch = 0; batch < tensor.shape().dim(0); batch++) {
                for (int row = 0; row < tensor.shape().dim(1); row++) {
                    for (int col = 0; col < tensor.shape().dim(2); col++) {
                        consumer.accept(tensor.get(batch, row, col));
                    }
                }
            }
            return;
        }
        throw new IllegalArgumentException("unsupported test tensor rank " + tensor.dims());
    }

    @FunctionalInterface
    private interface FloatPredicate {
        boolean test(float value);
    }

    @FunctionalInterface
    private interface FloatConsumer {
        void accept(float value);
    }
}
