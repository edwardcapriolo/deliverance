package net.deliverance.distributed;

import com.google.common.base.Preconditions;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.safetensors.Config;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

public class DeliveranceService {

    private static final long idealBillionParamsPerWorker = Integer.getInteger("jlama.ideal_b_params", 3);

    private static final int LAYER_IDX = 0;
    private static final int HEAD_IDX = 1;

    private static final Logger logger = LoggerFactory.getLogger(DeliveranceService.class);
    private final AbstractModel model;
    private final int workerCount;
    private final boolean splitHeads;
    private final boolean splitLayers;
    private final int headsPerLayerShard;
    private final int numHeadShards;
    private final int layersPerShard;
    private final int numLayerShards;
    private final List<List<Integer>> ordinalCombinations;
    private final ConcurrentMap<UUID, RegisterResponse> workers;
    private final ConcurrentMap<UUID, Runnable> discoveryActions;

    public DeliveranceService(AbstractModel model, int workerCount, boolean splitHeads, boolean splitLayers) {

        Preconditions.checkArgument(
                !splitHeads || splitLayers || workerCount <= model.getConfig().numberOfKeyValueHeads,
                "Worker count must be less than or equal to number of KV heads if not splitting layers"
        );
        this.model = model;
        this.workerCount = workerCount;
        this.splitHeads = splitHeads;
        this.splitLayers = splitLayers;
        this.workers = new ConcurrentHashMap<>();
        this.discoveryActions = new ConcurrentHashMap<>();
        //this.combinations = new ConcurrentHashMap<>();
        //this.generatorGroup = new GeneratorGroup();
        Config c = model.getConfig();

        int tmpHeadsPerLayerShard = splitHeads ? c.numberOfKeyValueHeads / workerCount : c.numberOfKeyValueHeads;
        int tmpLayersPerShard = splitLayers ? c.numberOfLayers / workerCount : c.numberOfLayers;

        if (splitLayers && splitHeads) {
            long queryParams = (long) c.embeddingLength * c.embeddingLength;
            long keyValueParams = 2L * c.numberOfKeyValueHeads * c.embeddingLength * c.embeddingLength;
            long attentionParams = queryParams + keyValueParams;
            long feedforwardParams = 2L * ((long) c.embeddingLength * c.hiddenLength + (long) c.hiddenLength * c.embeddingLength);
            long layerNormParams = 2L * c.embeddingLength;
            long paramsPerLayer = attentionParams + feedforwardParams + layerNormParams;

            long idealParamsPerWorker = idealBillionParamsPerWorker * 1_000_000_000L;
            long paramsPerWorker = tmpLayersPerShard * paramsPerLayer;

            if (paramsPerWorker > idealParamsPerWorker) {
                tmpHeadsPerLayerShard = Math.min(
                        Math.min(workerCount, c.numberOfKeyValueHeads),
                        (int) Math.ceilDivExact(paramsPerLayer, idealParamsPerWorker)
                );
                tmpHeadsPerLayerShard = nextPowerOfTwo(tmpHeadsPerLayerShard);
                tmpHeadsPerLayerShard = c.numberOfKeyValueHeads / tmpHeadsPerLayerShard;
                tmpLayersPerShard = tmpLayersPerShard * (c.numberOfKeyValueHeads / tmpHeadsPerLayerShard);
            } else {
                tmpHeadsPerLayerShard = c.numberOfKeyValueHeads;
            }
        }

        this.headsPerLayerShard = tmpHeadsPerLayerShard;
        this.numHeadShards = c.numberOfKeyValueHeads / headsPerLayerShard;
        this.layersPerShard = tmpLayersPerShard;
        this.numLayerShards = c.numberOfLayers / layersPerShard;

        logger.info("{} Layer Shards of {}, {} Head Shards of {}", numLayerShards, layersPerShard, numHeadShards, headsPerLayerShard);

        this.ordinalCombinations = new ArrayList<>(workerCount);
        for (int i = 0; i < numLayerShards; i++) {
            for (int j = 0; j < numHeadShards; j++) {
                ordinalCombinations.add(Arrays.asList( i, j ));
            }
        }
    }

    public static int nextPowerOfTwo(int n) {
        return n <= 1 ? 1 : Integer.highestOneBit(n - 1) * 2;
    }
    public static boolean isPowerOfTwoUsingBitwiseOperation(int n) {
        return (n != 0) && ((n & (n - 1)) == 0);
    }

    public int getHeadsPerLayerShard() {
        return headsPerLayerShard;
    }

    public int getNumHeadShards() {
        return numHeadShards;
    }

    public int getLayersPerShard() {
        return layersPerShard;
    }

    public int getNumLayerShards() {
        return numLayerShards;
    }

    public List<List<Integer>> getOrdinalCombinations() {
        return ordinalCombinations;
    }
}