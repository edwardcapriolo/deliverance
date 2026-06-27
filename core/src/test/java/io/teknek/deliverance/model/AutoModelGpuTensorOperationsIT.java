package io.teknek.deliverance.model;

import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

@Tag("large-model")
@Disabled("Manual GPU provider AutoModel smoke; requires local Qwen3 cache, Dawn/WebGPU, and native GPU libraries.")
public class AutoModelGpuTensorOperationsIT {

    @Test
    public void autoModelHydratesAvailableTensorOperations() {
        ModelFetcher fetch = new ModelFetcher("Qwen", "Qwen3-0.6B").withDownload(false);
        try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetch).buildLocalTransformerModel()) {
            assertTrue(model.tensorOperations(TensorProviderKind.GPU).isPresent());
            assertTrue(model.tensorOperations(TensorProviderKind.SIMD).isPresent());
            assertTrue(model.tensorOperations(TensorProviderKind.PANAMA).isPresent());
            assertTrue(model.tensorOperations(TensorProviderKind.NAIVE).isPresent());
            assertEquals("Native GPU Operations", model.tensorOperations(TensorProviderKind.GPU).orElseThrow().name());
        }
    }
}
