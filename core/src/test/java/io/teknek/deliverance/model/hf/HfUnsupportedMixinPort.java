package io.teknek.deliverance.model.hf;

import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

/** Documents inherited HF common tests whose APIs do not currently exist in Deliverance. */
public interface HfUnsupportedMixinPort {
    @Test
    @Disabled("HF CausalLMModelTest classification/QA inherited tests require non-causal-LM heads")
    default void hfCausalLmTaskHeadsUnsupported() {
    }

    @Test
    @Disabled("HF PipelineTesterMixin tests target transformers pipeline integration; Deliverance does not implement HF pipelines")
    default void hfPipelineTesterMixinUnsupported() {
    }

    @Test
    @Disabled("HF TrainingTesterMixin tests require autograd/training APIs; Deliverance model tests are inference-only")
    default void hfTrainingTesterMixinUnsupported() {
    }

    @Test
    @Disabled("HF TensorParallelTesterMixin tests target transformers tp_plan/device mesh APIs; Deliverance TP uses separate runtime transport APIs")
    default void hfTensorParallelTesterMixinUnsupported() {
    }

    @Test
    @Disabled("HF GenerationTesterMixin beam/sample/assistant/speculative/cache-implementation tests require HF generate API features not present in Deliverance generation")
    default void hfAdvancedGenerationTesterMixinUnsupported() {
    }

    @Test
    @Disabled("HF GenerationTesterMixin left-padding compatibility requires public attention_mask/position_ids forward APIs; Deliverance generation currently derives positions internally")
    default void hfGenerationLeftPaddingCompatibilityUnsupported() {
    }

    @Test
    @Disabled("HF GenerationTesterMixin inputs_embeds generation requires prepare_inputs_for_generation(inputs_embeds); Deliverance Gemma4 PLE requires token-derived per-layer inputs")
    default void hfGenerationInputsEmbedsGenerationUnsupported() {
    }

    @Test
    @Disabled("HF ModelTesterMixin resize/tie/offload/meta-device tests require mutable torch module APIs; Deliverance loads safetensors checkpoints")
    default void hfMutableTorchModelTesterMixinUnsupported() {
    }

    @Test
    @Disabled("HF attention backend tests cover eager/sdpa/flash/flex dispatch; Deliverance models use Java attention implementations")
    default void hfAttentionBackendTesterMixinUnsupported() {
    }
}
