package io.teknek.deliverance.springai;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties("spring.ai.deliverance")
public class DeliveranceConnectionProperties {
    private String mode = "client";
    private String model;
    private String modelConfig;
    private String baseUrl = "http://localhost:8080";
    private String username;
    private String password;
    private boolean autoPull = true;
    private HuggingFace huggingface = new HuggingFace();

    public String getMode() {
        return mode;
    }

    public void setMode(String mode) {
        this.mode = mode;
    }

    public String getModel() {
        return model;
    }

    public void setModel(String model) {
        this.model = model;
    }

    public String getBaseUrl() {
        return baseUrl;
    }

    public String getModelConfig() {
        return modelConfig;
    }

    public void setModelConfig(String modelConfig) {
        this.modelConfig = modelConfig;
    }

    public void setBaseUrl(String baseUrl) {
        this.baseUrl = baseUrl;
    }

    public String getUsername() {
        return username;
    }

    public void setUsername(String username) {
        this.username = username;
    }

    public String getPassword() {
        return password;
    }

    public void setPassword(String password) {
        this.password = password;
    }

    public boolean isAutoPull() {
        return autoPull;
    }

    public void setAutoPull(boolean autoPull) {
        this.autoPull = autoPull;
    }

    public HuggingFace getHuggingface() {
        return huggingface;
    }

    public void setHuggingface(HuggingFace huggingface) {
        this.huggingface = huggingface;
    }

    public static class HuggingFace {
        private String token;

        public String getToken() {
            return token;
        }

        public void setToken(String token) {
            this.token = token;
        }
    }
}
