package io.teknek.deliverance.safetensors;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.google.common.collect.ImmutableMap;

import java.util.Map;


public class SafeTensorIndexPojo {

    public static final String MODEL_INDEX_JSON =  "model.safetensors.index.json";
    public static final String SINGLE_MODEL_NAME = "model.safetensors";
    private final Map<String, String> metadata;
    private final Map<String, String> weightFileMap;

    @JsonCreator
    public SafeTensorIndexPojo(@JsonProperty("metadata") Map<String, String> metadata,
                               @JsonProperty("weight_map") Map<String, String> weightFileMap) {
        this.metadata = ImmutableMap.copyOf(metadata);
        this.weightFileMap = ImmutableMap.copyOf(weightFileMap);
    }


    public Map<String, String> getMetadata() {
        return metadata;
    }

    public Map<String, String> getWeightFileMap() {
        return weightFileMap;
    }

}
