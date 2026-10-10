package io.teknek.deliverance.model.llama;

import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

/**
 * Port inventory for Hugging Face's {@code Mask4DTestHard}.
 *
 * Source: /ai-code/transformers/tests/models/llama/test_modeling_llama.py
 *
 * The upstream tests require custom 4D attention masks, explicit position IDs, and Hugging Face StaticCache
 * objects. Deliverance does not expose those APIs through the equivalent Llama test harness yet.
 */
public class Mask4DTestHard {
    @Test
    @Disabled("HF test_stacked_causal_mask requires public attention_mask and position_ids forward APIs")
    public void test_stacked_causal_mask() {
    }

    @Test
    @Disabled("HF test_partial_stacked_causal_mask requires public 4D attention_mask, position_ids, and past_key_values APIs")
    public void test_partial_stacked_causal_mask() {
    }

    @Test
    @Disabled("HF test_stacked_causal_mask_static_cache requires the Hugging Face StaticCache API")
    public void test_stacked_causal_mask_static_cache() {
    }

    @Test
    @Disabled("HF test_partial_stacked_causal_mask_static_cache requires the Hugging Face StaticCache API")
    public void test_partial_stacked_causal_mask_static_cache() {
    }
}
