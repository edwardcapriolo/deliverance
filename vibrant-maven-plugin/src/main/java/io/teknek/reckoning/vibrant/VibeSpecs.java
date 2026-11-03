package io.teknek.reckoning.vibrant;

import org.apache.maven.plugins.annotations.Parameter;

import java.util.ArrayList;
import java.util.List;

public class VibeSpecs {
    @Parameter(property = "vibeSpec")
    private List<VibeSpec> vibeSpec = new ArrayList<>();

    public List<VibeSpec> getVibeSpec() {
        return vibeSpec;
    }

    public void setVibeSpec(List<VibeSpec> vibeSpec) {
        this.vibeSpec = vibeSpec;
    }

    @Override
    public String toString() {
        return "VibeSpecs{" +
                "vibeSpec=" + vibeSpec +
                '}';
    }
}
