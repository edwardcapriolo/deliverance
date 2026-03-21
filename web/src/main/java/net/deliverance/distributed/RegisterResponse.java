package net.deliverance.distributed;

public class RegisterResponse {
    String hostname;
    int peerPort;
    int modelShard;
    int numModelShards;
    int layerShard;
    int numLayerShards;
    int workerOrd;

    public String getHostname() {
        return hostname;
    }

    public void setHostname(String hostname) {
        this.hostname = hostname;
    }

    public int getPeerPort() {
        return peerPort;
    }

    public void setPeerPort(int peerPort) {
        this.peerPort = peerPort;
    }

    public int getModelShard() {
        return modelShard;
    }

    public void setModelShard(int modelShard) {
        this.modelShard = modelShard;
    }

    public int getNumModelShards() {
        return numModelShards;
    }

    public void setNumModelShards(int numModelShards) {
        this.numModelShards = numModelShards;
    }

    public int getLayerShard() {
        return layerShard;
    }

    public void setLayerShard(int layerShard) {
        this.layerShard = layerShard;
    }

    public int getNumLayerShards() {
        return numLayerShards;
    }

    public void setNumLayerShards(int numLayerShards) {
        this.numLayerShards = numLayerShards;
    }

    public int getWorkerOrd() {
        return workerOrd;
    }

    public void setWorkerOrd(int workerOrd) {
        this.workerOrd = workerOrd;
    }
}
