package io.teknek.deliverance.model.llama;

import io.teknek.deliverance.model.hf.HfUnsupportedMixinPort;

/**
 * Port marker for Hugging Face's {@code LlamaModelTest}.
 *
 * Source: /ai-code/transformers/tests/models/llama/test_modeling_llama.py
 *
 * Hugging Face's explicit coverage for this class is inherited from CausalLMModelTest and ModelTesterMixin.
 * The inherited test inventory is represented by the shared Deliverance HF mixin port and its disabled methods;
 * this class preserves the upstream class name for test inventory and future runnable ports.
 */
public class LlamaModelTest implements HfUnsupportedMixinPort {
}
