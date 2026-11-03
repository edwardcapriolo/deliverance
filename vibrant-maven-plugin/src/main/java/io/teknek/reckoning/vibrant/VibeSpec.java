package io.teknek.reckoning.vibrant;

import org.apache.maven.plugins.annotations.Parameter;

import java.util.List;

public class VibeSpec {

    @Parameter(name = "id")
    String id;

    @Parameter(name = "userMessage")
    List<String> userMessages;

    @Parameter(name = "systemMessage")
    List<String> systemMessages;

    @Parameter(name = "enabled", defaultValue = "true")
    boolean enabled;

    @Parameter(name = "generateTo", defaultValue = "generated-source")
    private String generateTo;

    @Parameter(name ="overwrite", defaultValue = "false")
    private boolean overwrite;

    @Parameter(name = "messageTransformer", defaultValue = "io.teknek.reckoning.vibrant.JavaResponseTransformer")
    private String messageTransformer;

    public VibeSpec(){

    }

    public String getId() {
        return id;
    }
    public void setId(){

    }
    public void setId(String id) {
        this.id = id;
    }

    public List<String> getUserMessages() {
        return userMessages;
    }

    public void setUserMessages(List<String> userMessages) {
        this.userMessages = userMessages;
    }

    public boolean isEnabled() {
        return enabled;
    }

    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
    }

    public String getGenerateTo() {
        return generateTo;
    }

    public void setGenerateTo(String generateTo) {
        this.generateTo = generateTo;
    }

    public List<String> getSystemMessages() {
        return systemMessages;
    }

    public void setSystemMessages(List<String> systemMessages) {
        this.systemMessages = systemMessages;
    }

    public boolean isOverwrite() {
        return overwrite;
    }

    public void setOverwrite(boolean overwrite) {
        this.overwrite = overwrite;
    }

    public String getMessageTransformer() {
        return messageTransformer;
    }

    public void setMessageTransformer(String messageTransformer) {
        this.messageTransformer = messageTransformer;
    }

    @Override
    public String toString() {
        return "VibeSpec{" +
                "id='" + id + '\'' +
                ", userMessages=" + userMessages +
                ", systemMessages=" + systemMessages +
                ", enabled=" + enabled +
                ", generateTo='" + generateTo + '\'' +
                '}';
    }
}
