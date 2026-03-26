package io.teknek.deliverance;

import io.teknek.deliverance.embedding.PoolingType;

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
