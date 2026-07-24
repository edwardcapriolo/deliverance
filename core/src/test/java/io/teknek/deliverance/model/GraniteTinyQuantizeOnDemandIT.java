package io.teknek.deliverance.model;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.safetensors.ModelQuantizer;
import io.teknek.deliverance.safetensors.SafeTensorIndexPojo;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertTrue;

@Tag("large-model")
class GraniteTinyQuantizeOnDemandIT {

    @Test
    //@Disabled("One-shot local setup: downloads ibm-granite/granite-4.0-h-tiny and generates ibm-granite/granite-4.0-h-tiny-JQ4")
    void fetchesGraniteTinyAndCreatesJq4WithQuantizeOnDemand() {
        ModelFetcher resolved = AutoModelForCausaLm.newBuilder(new ModelFetcher("ibm-granite", "granite-4.0-h-tiny"))
                .withQuantizeOnDemand(DType.Q4, "ibm-granite", "granite-4.0-h-tiny-JQ4")
                .resolveModelFetcherForLoad();

        Path target = resolved.pathForModel();
        assertTrue(Files.isDirectory(target), "Missing quantized target directory: " + target);
        assertTrue(Files.exists(target.resolve(".finished")), "Missing finished marker: " + target);
        assertTrue(Files.exists(target.resolve(ModelQuantizer.QUANTIZATION_MANIFEST)),
                "Missing quantization manifest: " + target);
        assertTrue(Files.exists(target.resolve(SafeTensorIndexPojo.MODEL_INDEX_JSON))
                        || Files.exists(target.resolve(SafeTensorIndexPojo.SINGLE_MODEL_NAME)),
                "Missing quantized safetensors weights: " + target);
    }
}
