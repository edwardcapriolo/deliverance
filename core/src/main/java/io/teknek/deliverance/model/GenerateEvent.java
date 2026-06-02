package io.teknek.deliverance.model;

public interface GenerateEvent {
    /**
     *
     * @param next the token returned from the inference pipeline
     * @param nextRaw decoded by the tokenizer
     * @param nextCleaned legacy alias for nextRaw; token rendering is handled by tokenizer decode
     * @param timing cumulative time to create this event from start of generate
     */
    void emit(int next, String nextRaw, String nextCleaned, float timing);
}
