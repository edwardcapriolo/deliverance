package io.teknek.deliverance.safetensors;

import com.fasterxml.jackson.databind.type.MapType;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorInfo;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

import static io.teknek.deliverance.safetensors.Weights.findDType;

/** Weight loader for one safetensors shard, intended for bounded-memory batch jobs such as quantization. */
public final class SafetensorsShardWeightLoader implements WeightLoader {
    private static final MapType METADATA_TYPE = JsonUtils.om.getTypeFactory()
            .constructMapType(Map.class, String.class, String.class);

    private final RandomAccessFile file;
    private final Map<String, String> metadata;
    private final Map<String, TensorInfo> tensorInfoMap;
    private final Weights weights;
    private final DType modelDType;

    public SafetensorsShardWeightLoader(Path shardFile) {
        try {
            this.file = new RandomAccessFile(shardFile.toFile(), "r");
            ByteBuffer header = file.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, Math.min(1 << 20, file.length()));
            this.metadata = new HashMap<>();
            this.tensorInfoMap = DefaultWeightLoader.readTensorInfoMap(header, Optional.of(metadata));
            int endOfHeaderPosition = header.position();
            ByteBuffer payload = file.getChannel().map(FileChannel.MapMode.READ_ONLY,
                    endOfHeaderPosition, file.length() - endOfHeaderPosition).order(ByteOrder.LITTLE_ENDIAN);
            this.modelDType = findDType(tensorInfoMap);
            this.weights = new Weights(metadata, tensorInfoMap, payload, Optional.of(this));
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
        return tensorInfoMap;
    }

    @Override
    public AbstractTensor load(String name) {
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
