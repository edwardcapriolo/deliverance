package io.teknek.deliverance.safetensors;

import com.google.common.base.Preconditions;
import com.google.common.collect.BiMap;
import com.google.common.collect.ImmutableBiMap;
import com.google.common.io.Files;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.model.DistributedContext;
import io.teknek.deliverance.tensor.TensorCache;

import java.io.File;
import java.util.List;
import java.util.Map;
import java.util.Optional;

public class Config {
    public final int contextLength;
    public final int embeddingLength;
    public final int attentionLength;
    public final int hiddenLength;
    public final int numberOfHeads;
    public final int numberOfKeyValueHeads;
    public final int headSize;
    public final ActivationFunction.Type activationFunction;
    public final int headGroupSize;
    public final int kvLength;
    public final boolean isGQA;
    public final int numberOfLayers;
    public final float layerNormEps;
    public final Float finalLogitSoftCapping;
    public final Float attnLogitSoftCapping;
    public final Float residualMultiplier;
    public final Float attentionMultiplier;
    public final Float embeddingMultiplier;
    public final Float logitMultiplier;
    public final int vocabularySize;
    public final int bosToken;
    public final List<Integer> eosTokens;
    //public final Optional<float[][]> ropeFreqs;
    public final Optional<BiMap<String, Integer>> classifcationLabels;

    private volatile DistributedContext dctx;
    private volatile File workingDirectory;

    public final TensorCache tensorCache;

    public Config(
            int contextLength,
            int embeddingLength,
            int hiddenLength,
            int numberOfHeads,
            int numberOfKeyValueHeads,
            int numberOfLayers,
            float layerNormEps,
            int vocabularySize,
            int bosToken,
            List<Integer> eosToken,
            ActivationFunction.Type activationFunction,
            Double ropeFreqsTheta,
            Double ropeScalingFactor,
            Integer headSize,
            Float attnLogitSoftCapping,
            Float finalLogitSoftCapping
    ) {
        this(
                contextLength,
                embeddingLength,
                hiddenLength,
                numberOfHeads,
                numberOfKeyValueHeads,
                numberOfLayers,
                layerNormEps,
                vocabularySize,
                bosToken,
                eosToken,
                activationFunction,
                ropeFreqsTheta,
                ropeScalingFactor,
                null,
                headSize == null ? embeddingLength / numberOfHeads : headSize,
                attnLogitSoftCapping,
                finalLogitSoftCapping,
                null,
                null,
                null,
                null
        );
    }

    public Config(
            int contextLength,
            int embeddingLength,
            int hiddenLength,
            int numberOfHeads,
            int numberOfKeyValueHeads,
            int numberOfLayers,
            float layerNormEps,
            int vocabularySize,
            int bosToken,
            List<Integer> eosToken,
            ActivationFunction.Type activationFunction,
            Double ropeFreqsTheta,
            Double ropeScalingFactor
    ) {
        this(
                contextLength,
                embeddingLength,
                hiddenLength,
                numberOfHeads,
                numberOfKeyValueHeads,
                numberOfLayers,
                layerNormEps,
                vocabularySize,
                bosToken,
                eosToken,
                activationFunction,
                ropeFreqsTheta,
                ropeScalingFactor,
                null,
                embeddingLength / numberOfHeads,
                null,
                null,
                null,
                null,
                null,
                null
        );
    }

    public Config(
            int contextLength,
            int embeddingLength,
            int hiddenLength,
            int numberOfHeads,
            int numberOfKeyValueHeads,
            int numberOfLayers,
            float layerNormEps,
            int vocabularySize,
            int bosToken,
            List<Integer> eosToken,
            ActivationFunction.Type activationFunction,
            Double ropeFreqsTheta,
            Double ropeScalingFactor,
            Float residualMultiplier,
            Float attentionMultiplier,
            Float embeddingMultiplier,
            Float logitMultiplier
    ) {
        this(
                contextLength,
                embeddingLength,
                hiddenLength,
                numberOfHeads,
                numberOfKeyValueHeads,
                numberOfLayers,
                layerNormEps,
                vocabularySize,
                bosToken,
                eosToken,
                activationFunction,
                ropeFreqsTheta,
                ropeScalingFactor,
                null,
                embeddingLength / numberOfHeads,
                null,
                null,
                residualMultiplier,
                attentionMultiplier,
                embeddingMultiplier,
                logitMultiplier
        );
    }

    public Config(
            int contextLength,
            int embeddingLength,
            int hiddenLength,
            int numberOfHeads,
            int numberOfKeyValueHeads,
            int numberOfLayers,
            float layerNormEps,
            int vocabularySize,
            int bosToken,
            List<Integer> eosToken,
            ActivationFunction.Type activationFunction,
            Double ropeFreqsTheta,
            Double ropeScalingFactor,
            Map<String, Integer> classifcationLabels
    ) {
        this(
                contextLength,
                embeddingLength,
                hiddenLength,
                numberOfHeads,
                numberOfKeyValueHeads,
                numberOfLayers,
                layerNormEps,
                vocabularySize,
                bosToken,
                eosToken,
                activationFunction,
                ropeFreqsTheta,
                ropeScalingFactor,
                classifcationLabels,
                embeddingLength / numberOfHeads,
                null,
                null,
                null,
                null,
                null,
                null
        );
    }

    public Config(
            int contextLength,
            int embeddingLength,
            int hiddenLength,
            int numberOfHeads,
            int numberOfKeyValueHeads,
            int numberOfLayers,
            float layerNormEps,
            int vocabularySize,
            int bosToken,
            List<Integer> eosTokens,
            ActivationFunction.Type activationFunction,
            Double ropeFreqsTheta,
            Double ropeScalingFactor,
            Map<String, Integer> classifcationLabels,
            Integer headSize,
            Float finalLogitSoftCapping,
            Float attnLogitSoftCapping,
            Float residualMultiplier,
            Float attentionMultiplier,
            Float embeddingMultiplier,
            Float logitMultiplier
    ) {
        this.contextLength = contextLength;
        this.attentionLength = numberOfHeads * headSize;
        this.embeddingLength = embeddingLength;
        this.hiddenLength = hiddenLength;
        this.numberOfHeads = numberOfHeads;
        this.numberOfKeyValueHeads = numberOfKeyValueHeads;
        this.numberOfLayers = numberOfLayers;
        this.layerNormEps = layerNormEps;
        this.vocabularySize = vocabularySize;
        this.bosToken = bosToken;
        this.eosTokens = eosTokens;
        this.tensorCache = TensorCache.instance;
        this.headSize = headSize;
        this.headGroupSize = numberOfHeads / numberOfKeyValueHeads;
        this.kvLength = numberOfKeyValueHeads * headSize;
        this.isGQA = numberOfKeyValueHeads < numberOfHeads;
        this.activationFunction = activationFunction;
       /*
        this.ropeFreqs = ropeFreqsTheta == null
                ? Optional.empty()
                : Optional.of(
                VectorMath.precomputeFreqsCis(headSize, contextLength, ropeFreqsTheta, ropeScalingFactor == null ? 1.0 : ropeScalingFactor)
        );*/

        this.classifcationLabels = classifcationLabels == null ? Optional.empty() : Optional.of(ImmutableBiMap.copyOf(classifcationLabels));

        this.finalLogitSoftCapping = finalLogitSoftCapping;
        this.attnLogitSoftCapping = attnLogitSoftCapping;
        this.residualMultiplier = residualMultiplier;
        this.attentionMultiplier = attentionMultiplier;
        this.embeddingMultiplier = embeddingMultiplier;
        this.logitMultiplier = logitMultiplier;

        // Set default values
        this.dctx = DistributedContext.builder(this).build();
    }

    /*
    public void setDistributedContext(DistributedContext dctx) {
        this.dctx = dctx;
    }*/

    public void setWorkingDirectory(File workingDirectory) {
        if (workingDirectory == null) {
            this.workingDirectory = Files.createTempDir();
            this.workingDirectory.deleteOnExit();
        } else {
            Preconditions.checkArgument(workingDirectory.isDirectory());
            this.workingDirectory = workingDirectory;
        }
    }

    public Optional<File> workingDirectory() {
        return Optional.ofNullable(this.workingDirectory);
    }


    public DistributedContext dctx() {
        return dctx;
    }

    public int maybeMapToGroupHead(int head) {
        if (!isGQA) return head;
        return Math.floorDiv(head, headGroupSize);
    }

    public boolean isClassifier() {
        return classifcationLabels.isPresent();
    }
}