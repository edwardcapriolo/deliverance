package io.teknek.deliverance.safetensors;

import java.util.Optional;

/**
 * Translates between Deliverance's base-model tensor names (e.g. {@code
 * "model.layers.3.self_attn.q_proj.weight"}) and the HuggingFace PEFT adapter tensor names
 * for the same module (e.g. {@code
 * "base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight"}).
 *
 * <p>The {@code base_model.model.} prefix (with the base tensor's own {@code model.}-prefixed
 * path kept intact after it, hence "model." appearing twice) and the {@code
 * .lora_A.weight"/".lora_B.weight"} suffixes are standard PEFT conventions, verified directly
 * against a real published adapter's safetensors header rather than assumed.</p>
 */
final class LoraTensorNames {

    private static final String PEFT_PREFIX = "base_model.model.";
    private static final String LORA_A_SUFFIX = ".lora_A.weight";
    private static final String LORA_B_SUFFIX = ".lora_B.weight";
    private static final String WEIGHT_SUFFIX = ".weight";

    private LoraTensorNames() {
    }

    /**
     * Extracts the trailing module name that both Deliverance's base tensor names and a PEFT
     * adapter's {@code target_modules} list use to identify a target, e.g. {@code
     * "model.layers.3.self_attn.q_proj.weight"} -> {@code "q_proj"}.
     *
     * <p>Returns empty if {@code baseTensorName} doesn't have the expected {@code ".weight"}
     * suffix (e.g. a {@code .qb} quantization-block tensor), since such tensors are never
     * LoRA targets.</p>
     */
    static Optional<String> moduleSuffix(String baseTensorName) {
        String withoutWeight = stripWeightSuffix(baseTensorName);
        if (withoutWeight == null) {
            return Optional.empty();
        }
        int lastDot = withoutWeight.lastIndexOf('.');
        return Optional.of(lastDot < 0 ? withoutWeight : withoutWeight.substring(lastDot + 1));
    }

    static String loraA(String baseTensorName) {
        return adapterTensorName(baseTensorName, LORA_A_SUFFIX);
    }

    static String loraB(String baseTensorName) {
        return adapterTensorName(baseTensorName, LORA_B_SUFFIX);
    }

    private static String adapterTensorName(String baseTensorName, String loraSuffix) {
        String withoutWeight = stripWeightSuffix(baseTensorName);
        if (withoutWeight == null) {
            throw new IllegalArgumentException("Expected a base tensor name ending in \".weight\": " + baseTensorName);
        }
        return PEFT_PREFIX + withoutWeight + loraSuffix;
    }

    private static String stripWeightSuffix(String baseTensorName) {
        if (!baseTensorName.endsWith(WEIGHT_SUFFIX)) {
            return null;
        }
        return baseTensorName.substring(0, baseTensorName.length() - WEIGHT_SUFFIX.length());
    }
}
