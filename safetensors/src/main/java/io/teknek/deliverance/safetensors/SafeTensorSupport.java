package io.teknek.deliverance.safetensors;

import com.fasterxml.jackson.databind.JsonNode;


import java.io.File;
import java.io.IOException;
import java.io.UncheckedIOException;

public class SafeTensorSupport {
/*
    public static ModelType detectModel(File configFile) {
        JsonNode rootNode;
        try {
            rootNode = JsonUtils.om.readTree(configFile);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        if (!rootNode.has("model_type")) {
            throw new IllegalArgumentException("Config missing model_type field.");
        }
        return ModelSupport.getModelType(rootNode.get("model_type").textValue().toUpperCase());
    }
*/
}
