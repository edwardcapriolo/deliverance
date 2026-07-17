package io.teknek.deliverance.safetensors;

import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class LoraTensorNamesTest {

    @Test
    void moduleSuffixExtractsTrailingModuleName() {
        Map<String, String> baseTensorNameToExpectedModule = Map.of(
                "model.layers.3.self_attn.q_proj.weight", "q_proj",
                "model.layers.3.self_attn.k_proj.weight", "k_proj",
                "model.layers.3.self_attn.v_proj.weight", "v_proj",
                "model.layers.3.self_attn.o_proj.weight", "o_proj",
                "model.layers.0.mlp.gate_proj.weight", "gate_proj",
                "model.layers.0.mlp.up_proj.weight", "up_proj",
                "model.layers.0.mlp.down_proj.weight", "down_proj",
                "model.embed_tokens.weight", "embed_tokens",
                "lm_head.weight", "lm_head"
        );
        baseTensorNameToExpectedModule.forEach((baseTensorName, expectedModule) ->
                assertEquals(Optional.of(expectedModule), LoraTensorNames.moduleSuffix(baseTensorName), baseTensorName));
    }

    @Test
    void moduleSuffixIsEmptyForNonWeightTensors() {
        assertEquals(Optional.empty(), LoraTensorNames.moduleSuffix("model.layers.0.self_attn.q_proj.weight.qb"));
        assertEquals(Optional.empty(), LoraTensorNames.moduleSuffix("model.layers.0.self_attn.q_proj"));
    }

    @Test
    void loraAAndLoraBFollowVerifiedPeftNamingConvention() {
        // Verified directly against a real published adapter's safetensors header
        // (bunnycore/Llama-3.2-1b-chatml-lora_model): "base_model.model." + the base tensor's
        // own "model."-prefixed path, with ".weight" replaced by ".lora_A.weight"/".lora_B.weight".
        String base = "model.layers.0.self_attn.q_proj.weight";
        assertEquals("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight", LoraTensorNames.loraA(base));
        assertEquals("base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight", LoraTensorNames.loraB(base));
    }

    @Test
    void loraANameRejectsNonWeightBaseTensorNames() {
        assertThrows(IllegalArgumentException.class, () -> LoraTensorNames.loraA("model.layers.0.self_attn.q_proj"));
        assertThrows(IllegalArgumentException.class, () -> LoraTensorNames.loraB("model.layers.0.self_attn.q_proj"));
    }

    @Test
    void moduleSuffixRoundTripsThroughLoraNaming() {
        // The module name extracted from a base tensor name should be exactly what a real
        // adapter_config.json's target_modules entries look like (bare module names, no
        // "self_attn."/"mlp." prefix, no PEFT wrapping) -- this is the join key between
        // LoraAdapterConfig.targetModules and LoraAdapter.deltaFor(baseTensorName).
        String base = "model.layers.5.self_attn.o_proj.weight";
        Optional<String> module = LoraTensorNames.moduleSuffix(base);
        assertTrue(module.isPresent());
        assertEquals("o_proj", module.get());
    }
}
