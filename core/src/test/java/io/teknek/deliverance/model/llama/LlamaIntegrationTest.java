package io.teknek.deliverance.model.llama;

import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

/**
 * Port inventory for Hugging Face's {@code LlamaIntegrationTest}.
 *
 * Source: /ai-code/transformers/tests/models/llama/test_modeling_llama.py
 *
 * These tests remain present by upstream name. They are disabled because the upstream cases require
 * PyTorch, gated 7B/8B checkpoints, accelerator-specific execution, or torch compile/export APIs.
 */
public class LlamaIntegrationTest {
    @Test
    @Disabled("HF test_llama_3_1_hard requires the gated Meta-Llama-3.1-8B-Instruct checkpoint and accelerator-specific PyTorch execution")
    public void test_llama_3_1_hard() {
    }

    @Test
    @Disabled("HF test_model_7b_logits_bf16 requires the gated Llama-2-7b checkpoint and PyTorch accelerator logits")
    public void test_model_7b_logits_bf16() {
    }

    @Test
    @Disabled("HF test_model_7b_logits requires the gated Llama-2-7b checkpoint and PyTorch accelerator logits")
    public void test_model_7b_logits() {
    }

    @Test
    @Disabled("HF test_compile_static_cache requires torch.compile and the Hugging Face StaticCache generation API")
    public void test_compile_static_cache() {
    }

    @Test
    @Disabled("HF test_export_static_cache requires torch.export/ExecuTorch integration")
    public void test_export_static_cache() {
    }
}
