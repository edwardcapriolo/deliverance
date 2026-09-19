package io.teknek.deliverance.tensorlib;

import io.teknek.deliverance.tensor.TensorShape;

import java.util.List;
import java.util.Map;

/** Model-scoped adaptive split selection for explicitly opted-in TensorPlan nodes. */
public interface TensorPlanAdaptiveSplitTuner {
    int chooseSplit(String planName, TensorShape shape, int defaultSplit, int minSplit, int maxSplit);

    void observeSplit(String planName, TensorShape shape, int split, long elapsedNanos);

    String chooseAlternate(String planName, List<String> candidates);

    void observeAlternate(String planName, String candidate, long elapsedNanos);

    default void observeAlternate(String planName, String candidate, long elapsedNanos, Map<String, String> tags) {
        observeAlternate(planName, candidate, elapsedNanos);
    }
}
