package io.teknek.reckoning.integration.granite;

import io.teknek.reckoning.model.AbstractModel;
import io.teknek.reckoning.model.AutoModelForCausaLm;
import io.teknek.reckoning.safetensors.fetch.ModelFetcher;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.io.File;

import static org.junit.jupiter.api.Assertions.assertTrue;

/** Red porting test for the Granite 4.1 3B LLMON checkpoint. */
@Tag("large-model")
@Tag("model-port-red")
class Granite41LlmonPreviewPortIT {

    @Test
    void downloadsCheckpointThenStopsAtUnsupportedGraniteDispatch() throws Exception {
        ModelFetcher fetcher = new ModelFetcher("ibm-granite", "granite-4.1-3b-llmon-preview")
                .withDownload(true);
        File modelDirectory = fetcher.maybeDownload();
        assertTrue(new File(modelDirectory, "config.json").isFile(),
                "Downloaded model must contain config.json: " + modelDirectory);

        // Deliberately red until the Granite LLMON Config and model mapping are implemented.
        try (AbstractModel ignored = AutoModelForCausaLm.newBuilder(fetcher).buildLocalTransformerModel()) {
            throw new AssertionError("Granite 4.1 LLMON port unexpectedly loaded");
        }
    }

}
