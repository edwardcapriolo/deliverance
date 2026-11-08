package io.teknek.reckoning.safetensors;

import com.google.common.collect.ImmutableMap;
import com.google.common.primitives.Ints;
import io.teknek.reckoning.DType;
import io.teknek.reckoning.tensor.TensorInfo;
import io.teknek.reckoning.tensor.TensorShape;
import io.teknek.reckoning.tensor2.Lighter;
import io.teknek.reckoning.tensor2.Q4Layout;
import io.teknek.reckoning.tensor2.TensorRef;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.*;

public class Weights implements WeightLoader {
    private final Map<String, String> metadata;
    private final Map<String, TensorInfo> tensorInfoMap;
    private final ByteBuffer bytes;
    private final DType majorityDType;
    private final Optional<WeightLoader> parent;
    private Lighter lighter;


    Weights(Map<String, String> metadata, Map<String, TensorInfo> tensorInfoMap, ByteBuffer bytes, Optional<WeightLoader> parent) {
        this.metadata = ImmutableMap.copyOf(metadata);
        this.tensorInfoMap = ImmutableMap.copyOf(tensorInfoMap);
        this.bytes = bytes.duplicate();
        this.majorityDType = findDType(tensorInfoMap);
        this.parent = parent;
    }

    public static DType findDType(Map<String, TensorInfo> tensorInfoMap) {
        EnumMap<DType, Integer> counts = new EnumMap<>(DType.class);
        for (Map.Entry<String, TensorInfo> e : tensorInfoMap.entrySet()) {
            if (!e.getKey().endsWith(".qb")) {
                counts.put(e.getValue().dType, counts.getOrDefault(e.getValue().dType, 0) + 1);
            }
        }

        int max = 0;
        DType maxType = null;
        for (Map.Entry<DType, Integer> e : counts.entrySet()) {
            if (e.getValue() > max) {
                max = e.getValue();
                maxType = e.getKey();
            }
        }

        // FIXME don't really support F16 atm
        return maxType == DType.F16 ? DType.F32 : maxType;
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
    public TensorRef load(String name) {
        return load(name, requireLighter(), parent.orElse(this));
    }

    @Override
    public void setLighter(Lighter lighter) {
        this.lighter = Objects.requireNonNull(lighter, "lighter");
    }

    TensorRef load(String name, Lighter lighter, WeightLoader sidecarLoader) {
        TensorInfo info = tensorInfoMap.get(name);
        if (info == null) {
            throw new NoSuchElementException(name + " not found in weights");
        }
        int payloadOffset = Ints.checkedCast(info.dataOffsets[0]);
        int payloadLength = Ints.checkedCast(info.dataOffsets[1] - info.dataOffsets[0]);
        TensorShape shape = toTensorShape(info.shape);
        Map<String, TensorRef> sidecars = Map.of();
        if (info.dType == DType.Q4 || info.dType == DType.I8) {
            String sidecarName = info.dType == DType.Q4 ? "q4.scale" : "q8.scale";
            sidecars = Map.of(sidecarName, sidecarLoader.loadRef(name + ".qb"));
        }
        return TensorRef.mapped(this.bytes, payloadOffset, payloadLength, shape, info.dType, stride(shape), sidecars);
    }

    @Override
    public TensorRef loadRows(String name, int rowOffset, int rowCount) {
        return load(name, new TensorShardSpec(TensorShardAxis.ROWS, rowOffset, rowOffset + rowCount),
                requireLighter(), parent.orElse(this));
    }

    @Override
    public TensorRef load(String name, TensorShardSpec shardSpec) {
        return load(name, shardSpec, requireLighter(), parent.orElse(this));
    }

    private Lighter requireLighter() {
        if (lighter == null) {
            throw new IllegalStateException("TensorRef weight loading requires a configured Lighter");
        }
        return lighter;
    }

    private static int stride(TensorShape shape) {
        return shape.first() > 1 && shape.dims() == 2 ? Math.toIntExact(shape.getOffset(1, 0)) : 0;
    }

    TensorRef load(String name, TensorShardSpec shardSpec, Lighter lighter, WeightLoader sidecarLoader) {
        TensorInfo info = tensorInfoMap.get(name);
        if (info == null) {
            throw new NoSuchElementException(name + " not found in weights");
        }
        if (info.shape.length != 2) {
            throw new IllegalArgumentException("Tensor sharding only supported for 2D tensors: " + name);
        }

        int rows = Ints.checkedCast(info.shape[0]);
        int cols = Ints.checkedCast(info.shape[1]);
        ByteBuffer payload;
        TensorShape shape;
        TensorShardSpec sidecarSpec = shardSpec;
        if (shardSpec.axis() == TensorShardAxis.ROWS) {
            validateRange(shardSpec, rows, name);
            int bytesPerRow = rawBytesPerRow(info.dType, cols, name);
            payload = slicePayload(info.dataOffsets[0] + (long) shardSpec.startInclusive() * bytesPerRow,
                    shardSpec.length() * bytesPerRow);
            shape = TensorShape.of(shardSpec.length(), cols);
        } else {
            validateRange(shardSpec, cols, name);
            if (info.dType == DType.I8) {
                throw new UnsupportedOperationException("Column sharding for I8 tensors is not supported: " + name);
            }
            int bytesPerColumn = info.dType == DType.Q4 ? 0 : info.dType.size();
            int sourceBytesPerRow = info.dType == DType.Q4 ? q4BytesPerRow(cols, name) : cols * bytesPerColumn;
            int shardBytesPerRow = info.dType == DType.Q4
                    ? q4BytesPerRow(shardSpec.length(), name)
                    : shardSpec.length() * bytesPerColumn;
            if (info.dType == DType.Q4) {
                validateQ4ColumnRange(shardSpec, name);
                sidecarSpec = new TensorShardSpec(TensorShardAxis.COLUMNS,
                        shardSpec.startInclusive() / Q4Layout.BLOCK_SIZE,
                        shardSpec.endExclusive() / Q4Layout.BLOCK_SIZE);
            }
            payload = ByteBuffer.allocateDirect(rows * shardBytesPerRow).order(ByteOrder.LITTLE_ENDIAN);
            for (int row = 0; row < rows; row++) {
                long sourceOffset = info.dataOffsets[0] + (long) row * sourceBytesPerRow
                        + (info.dType == DType.Q4 ? shardSpec.startInclusive() / 2
                                : (long) shardSpec.startInclusive() * bytesPerColumn);
                ByteBuffer source = slicePayload(sourceOffset, shardBytesPerRow);
                payload.put(source);
            }
            payload.flip();
            shape = TensorShape.of(rows, shardSpec.length());
        }

        TensorRef ref = lighter.allocate(info.dType, shape);
        try {
            lighter.copyFrom(payload, ref);
            if (info.dType == DType.Q4 || info.dType == DType.I8) {
                TensorRef sourceScale = sidecarLoader.loadRef(name + ".qb", sidecarSpec);
                try (sourceScale) {
                    lighter.copyScale(sourceScale, ref);
                }
            }
            return ref;
        } catch (RuntimeException e) {
            ref.close();
            throw e;
        }
    }

    private ByteBuffer slicePayload(long offset, int length) {
        return bytes.duplicate()
                .order(ByteOrder.LITTLE_ENDIAN)
                .position(Ints.checkedCast(offset))
                .limit(Ints.checkedCast(offset + length))
                .slice()
                .order(ByteOrder.LITTLE_ENDIAN);
    }

    private static int rawBytesPerRow(DType dType, int cols, String name) {
        if (dType == DType.Q4) {
            return q4BytesPerRow(cols, name);
        }
        return cols * dType.size();
    }

    private static int q4BytesPerRow(int cols, String name) {
        if (cols % Q4Layout.BLOCK_SIZE != 0) {
            throw new IllegalArgumentException("Q4 sharding requires column count aligned to "
                    + Q4Layout.BLOCK_SIZE + ": " + name);
        }
        return cols / 2;
    }

    private static void validateRange(TensorShardSpec shardSpec, int dimension, String name) {
        if (shardSpec.endExclusive() > dimension) {
            throw new IllegalArgumentException("Invalid " + shardSpec.axis().name().toLowerCase() + " range "
                    + shardSpec.startInclusive() + "," + shardSpec.endExclusive() + " for " + name);
        }
    }

    private static void validateQ4ColumnRange(TensorShardSpec shardSpec, String name) {
        if (shardSpec.startInclusive() % Q4Layout.BLOCK_SIZE != 0
                || shardSpec.endExclusive() % Q4Layout.BLOCK_SIZE != 0) {
            throw new IllegalArgumentException("Q4 column sharding requires shard range aligned to "
                    + Q4Layout.BLOCK_SIZE + ": " + name);
        }
    }

    /**
     * Converts safetensors shapes into Reckoning tensor shapes.
     *
     * Safetensors permits zero-dimensional scalar tensors with {@code shape: []}, while Reckoning
     * tensors are always represented as at least two-dimensional. Scalar tensors are therefore
     * normalized to a logical {@code [1,1]} tensor so they can still be loaded and consumed by the
     * rest of the stack.
     */
    static TensorShape toTensorShape(int[] shape) {
        if (shape.length == 0) {
            return TensorShape.of(1, 1);
        }
        return TensorShape.of(shape);
    }

    @Override
    public DType getModelDType() {
        return majorityDType;
    }

    @Override
    public String toString() {
        return "SafeTensor{" + "metadata=" + metadata + ", tensorInfoMap=" + tensorInfoMap + ", bytes=" + bytes + '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Weights weights = (Weights) o;
        return Objects.equals(metadata, weights.metadata) && Objects.equals(tensorInfoMap, weights.tensorInfoMap);
    }

    @Override
    public int hashCode() {
        return Objects.hash(metadata, tensorInfoMap);
    }

    @Override
    public void close() throws Exception {}
}
