package io.teknek.deliverance.model;

public interface GenerateEvent {
    /**
     *
     * @param next the token returned from the inference pipeline
     * @param nextRaw decoded by the tokenizer
     * @param nextCleaned nextRow furher altered by the toknizerRenderer
     * @param timing cumulative time to create this event from start of generate
     */
    void emit(int next, String nextRaw, String nextCleaned, float timing);
}
