package io.teknek.deliverance.safetensors;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorInfo;
import com.google.common.collect.ImmutableMap;
import com.google.common.primitives.Ints;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.List;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

import static io.teknek.deliverance.safetensors.Weights.findDType;

/** Weight loader for one safetensors shard, intended for bounded-memory batch jobs such as quantization. */
public final class SafetensorsShardWeightLoader implements WeightLoader {
    private final RandomAccessFile file;
    private final Map<String, String> metadata;
    private final Map<String, TensorInfo> allTensorInfoMap;
    private final Map<String, Weights> weightMap;
    private final DType modelDType;

    public SafetensorsShardWeightLoader(Path shardFile) {
        try {
            this.file = new RandomAccessFile(shardFile.toFile(), "r");
            ByteBuffer header = file.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, Math.min(1 << 20, file.length()));
            this.metadata = new HashMap<>();
            Map<String, TensorInfo> tensorInfoMap = DefaultWeightLoader.readTensorInfoMap(header, Optional.of(metadata));
            int endOfHeaderPosition = header.position();
            Map<List<Long>, List<String>> splits = DefaultWeightLoader.computeMmapSplits(tensorInfoMap, file.length());
            this.allTensorInfoMap = new ConcurrentHashMap<>(tensorInfoMap);
            this.weightMap = new ConcurrentHashMap<>();
            for (Map.Entry<List<Long>, List<String>> split : splits.entrySet()) {
                long offset = split.getKey().get(0);
                long length = split.getKey().get(1);
                List<String> tensors = split.getValue();
                int lengthInt = Ints.checkedCast(length - offset);
                ByteBuffer buf = file.getChannel().map(FileChannel.MapMode.READ_ONLY,
                        endOfHeaderPosition + offset, lengthInt);
                Map<String, TensorInfo> mmapTensorInfoMap = tensorInfoMap.entrySet()
                        .stream()
                        .filter(entry -> tensors.contains(entry.getKey()))
                        .collect(ImmutableMap.toImmutableMap(Map.Entry::getKey, Map.Entry::getValue));
                Weights mmapWeights = new Weights(metadata, mmapTensorInfoMap, buf, Optional.of(this));
                for (String tensor : tensors) {
                    weightMap.put(tensor, mmapWeights);
                }
            }
            this.modelDType = findDType(allTensorInfoMap);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    @Override
    public Map<String, String> metadata() {
        return metadata;
    }

    @Override
    public Map<String, TensorInfo> tensorInfoMap() {
        return allTensorInfoMap;
    }

    @Override
    public AbstractTensor load(String name) {
        Weights weights = weightMap.get(name);
        if (weights == null) {
            throw new RuntimeException("weight cant be found " + name + " list" + weightMap.keySet());
        }
        return weights.load(name);
    }

    @Override
    public DType getModelDType() {
        return modelDType;
    }

    @Override
    public void close() {
        try {
            file.close();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }
}
