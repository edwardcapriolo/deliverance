package io.teknek.reckoning;

import io.teknek.reckoning.embedding.PoolingType;

import java.util.Map;

public interface Classifier {

    /**
     * Classify a string
     *
     * @param input the input string
     * @return the classification (if supported)
     */
    Map<String, Float> classify(String input, PoolingType poolingType);
}
