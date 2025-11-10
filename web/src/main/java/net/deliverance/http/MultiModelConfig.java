package net.deliverance.http;

import java.util.Objects;

public class MultiModelConfig {
    private String modelName;
    private String modelOwner;
    private String inferenceType;

    public MultiModelConfig() {

    }

    public String getModelName() {
        return modelName;
    }

    public void setModelName(String modelName) {
        this.modelName = modelName;
    }

    public String getModelOwner() {
        return modelOwner;
    }

    public void setModelOwner(String modelOwner) {
        this.modelOwner = modelOwner;
    }

    public String getInferenceType() {
        return inferenceType;
    }

    public void setInferenceType(String inferenceType) {
        this.inferenceType = inferenceType;
    }

    @Override
    public boolean equals(Object o) {
        if (o == null || getClass() != o.getClass()) return false;
        MultiModelConfig that = (MultiModelConfig) o;
        return Objects.equals(modelName, that.modelName) && Objects.equals(modelOwner, that.modelOwner);
    }

    @Override
    public int hashCode() {
        return Objects.hash(modelName, modelOwner);
    }
}
