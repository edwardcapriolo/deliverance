package io.teknek.reckoning.model;

public class DoNothingGenerateEvent implements GenerateEvent {
    @Override
    public void emit(int next, String nextRaw, String nextCleaned, float timing) {

    }
}
