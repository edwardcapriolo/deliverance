package net.deliverance.http;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class MultiModelConfig {
    private String modelName;
    private String modelOwner;
    private String inferenceType;
    private TensorParallelConfig tensorParallel = new TensorParallelConfig();

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

    public TensorParallelConfig getTensorParallel() {
        return tensorParallel;
    }

    public void setTensorParallel(TensorParallelConfig tensorParallel) {
        this.tensorParallel = tensorParallel;
    }

    public static class TensorParallelConfig {
        private boolean enabled;
        private String cluster = "deliverance-tp-local";
        private String nodeId = "coordinator";
        private String uri = "udp://127.0.0.1:42606";
        private List<String> seeds = new ArrayList<>();
        private String deployment = "benchmark";
        private int size = 4;
        private int maxRanksPerWorker = 2;
        private int readyTimeoutSeconds = 120;
        private int rankEndpointTimeoutSeconds = 300;
        private String outputHeadQuantization = "Q4";

        public boolean isEnabled() {
            return enabled;
        }

        public void setEnabled(boolean enabled) {
            this.enabled = enabled;
        }

        public String getCluster() {
            return cluster;
        }

        public void setCluster(String cluster) {
            this.cluster = cluster;
        }

        public String getNodeId() {
            return nodeId;
        }

        public void setNodeId(String nodeId) {
            this.nodeId = nodeId;
        }

        public String getUri() {
            return uri;
        }

        public void setUri(String uri) {
            this.uri = uri;
        }

        public List<String> getSeeds() {
            return seeds;
        }

        public void setSeeds(List<String> seeds) {
            this.seeds = seeds;
        }

        public String getDeployment() {
            return deployment;
        }

        public void setDeployment(String deployment) {
            this.deployment = deployment;
        }

        public int getSize() {
            return size;
        }

        public void setSize(int size) {
            this.size = size;
        }

        public int getMaxRanksPerWorker() {
            return maxRanksPerWorker;
        }

        public void setMaxRanksPerWorker(int maxRanksPerWorker) {
            this.maxRanksPerWorker = maxRanksPerWorker;
        }

        public int getReadyTimeoutSeconds() {
            return readyTimeoutSeconds;
        }

        public void setReadyTimeoutSeconds(int readyTimeoutSeconds) {
            this.readyTimeoutSeconds = readyTimeoutSeconds;
        }

        public int getRankEndpointTimeoutSeconds() {
            return rankEndpointTimeoutSeconds;
        }

        public void setRankEndpointTimeoutSeconds(int rankEndpointTimeoutSeconds) {
            this.rankEndpointTimeoutSeconds = rankEndpointTimeoutSeconds;
        }

        public String getOutputHeadQuantization() {
            return outputHeadQuantization;
        }

        public void setOutputHeadQuantization(String outputHeadQuantization) {
            this.outputHeadQuantization = outputHeadQuantization;
        }
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
