package io.teknek.reckoning.model.hf;

import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

/** Documents inherited HF common tests whose APIs do not currently exist in Reckoning. */
public interface HfUnsupportedMixinPort {
    @Test
    @Disabled("HF CausalLMModelTest classification/QA inherited tests require non-causal-LM heads")
    default void hfCausalLmTaskHeadsUnsupported() {
    }

    @Test
    @Disabled("HF PipelineTesterMixin tests target transformers pipeline integration; Reckoning does not implement HF pipelines")
    default void hfPipelineTesterMixinUnsupported() {
    }

    @Test
    @Disabled("HF TrainingTesterMixin tests require autograd/training APIs; Reckoning model tests are inference-only")
    default void hfTrainingTesterMixinUnsupported() {
    }

    @Test
    @Disabled("HF TensorParallelTesterMixin tests target transformers tp_plan/device mesh APIs; Reckoning TP uses separate runtime transport APIs")
    default void hfTensorParallelTesterMixinUnsupported() {
    }

    @Test
    @Disabled("HF GenerationTesterMixin beam/sample/assistant/speculative/cache-implementation tests require HF generate API features not present in Reckoning generation")
    default void hfAdvancedGenerationTesterMixinUnsupported() {
    }

    @Test
    @Disabled("HF GenerationTesterMixin left-padding compatibility requires public attention_mask/position_ids forward APIs; Reckoning generation currently derives positions internally")
    default void hfGenerationLeftPaddingCompatibilityUnsupported() {
    }

    @Test
    @Disabled("HF GenerationTesterMixin inputs_embeds generation requires prepare_inputs_for_generation(inputs_embeds); Reckoning Gemma4 PLE requires token-derived per-layer inputs")
    default void hfGenerationInputsEmbedsGenerationUnsupported() {
    }

    @Test
    @Disabled("HF ModelTesterMixin resize/tie/offload/meta-device tests require mutable torch module APIs; Reckoning loads safetensors checkpoints")
    default void hfMutableTorchModelTesterMixinUnsupported() {
    }

    @Test
    @Disabled("HF attention backend tests cover eager/sdpa/flash/flex dispatch; Reckoning models use Java attention implementations")
    default void hfAttentionBackendTesterMixinUnsupported() {
    }
}
