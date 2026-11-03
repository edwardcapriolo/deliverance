package io.teknek.reckoning.vibrant;

import io.teknek.reckoning.DType;
import org.apache.maven.plugins.annotations.Parameter;

public class ModelConfig {
    //@Parameter(name="modelName" , defaultValue = "Llama-3.2-3B-Instruct-JQ4")
    private String modelName;

    //@Parameter(name= "owner", defaultValue = "tjake")
    private String owner;

    private String workingMemType;

    private String quantizedMemType;

    public ModelConfig(){
        setModelName("Llama-3.2-3B-Instruct-JQ4");
        setOwner("tjake");
        setWorkingMemType(DType.F32.toString());
        setQuantizedMemType(DType.I8.toString());
    }

    public String getModelName() {
        return modelName;
    }

    public void setModelName(String modelName) {
        this.modelName = modelName;
    }

    public String getOwner() {
        return owner;
    }

    public void setOwner(String owner) {
        this.owner = owner;
    }

    @Override
    public String toString() {
        return "ModelConfig{" +
                "modelName='" + modelName + '\'' +
                ", owner='" + owner + '\'' +
                '}';
    }

    public String getWorkingMemType() {
        return workingMemType;
    }

    public void setWorkingMemType(String workingMemType) {
        this.workingMemType = workingMemType;
    }

    public String getQuantizedMemType() {
        return quantizedMemType;
    }

    public void setQuantizedMemType(String quantizedMemType) {
        this.quantizedMemType = quantizedMemType;
    }
}
