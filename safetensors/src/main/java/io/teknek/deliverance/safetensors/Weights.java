package io.teknek.deliverance.safetensors;

import com.google.common.collect.ImmutableMap;
import com.google.common.primitives.Ints;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorInfo;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.impl.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.ShortBuffer;
import java.util.*;

public class Weights implements WeightLoader {
    private static final Logger logger = LoggerFactory.getLogger(Weights.class);
    private final Map<String, String> metadata;
    private final Map<String, TensorInfo> tensorInfoMap;
    private final ByteBuffer bytes;
    private final DType majorityDType;
    private final Optional<WeightLoader> parent;


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
    public AbstractTensor load(String name) {
        TensorInfo info = tensorInfoMap.get(name);
        if (info == null) {
            throw new NoSuchElementException(name + " not found in weights");
        }
        ByteBuffer b = bytes.duplicate()
                .order(ByteOrder.LITTLE_ENDIAN)
                .position(Ints.checkedCast(info.dataOffsets[0]))
                .limit(Ints.checkedCast(info.dataOffsets[1]));
        return loadTensorFromBuffer(name, info.dType, majorityDType, toTensorShape(info.shape), b, parent.orElse(this));
    }

    /**
     * Converts safetensors shapes into Deliverance tensor shapes.
     *
     * Safetensors permits zero-dimensional scalar tensors with {@code shape: []}, while Deliverance
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

    static AbstractTensor loadTensorFromBuffer(
            String name,
            DType dType,
            DType majorityDType,
            TensorShape shape,
            ByteBuffer b,
            WeightLoader loader
    ) {
        int len;
        FloatBuffer fb;
        ShortBuffer sb;
        AbstractTensor t;
        switch (dType) {
            case F32:
                fb = b.asFloatBuffer().slice();
                t = new FloatBufferTensor(name, fb, shape, true);
                break;
            case F16:
                // If the majority of the weights are F32 then convert to F32
                if (majorityDType == DType.F32) {
                    len = b.remaining() / DType.F16.size();
                    ByteBuffer bb = ByteBuffer.allocate(len * DType.F32.size()).order(ByteOrder.LITTLE_ENDIAN);
                    for (int i = 0; i < len * DType.F32.size(); i += DType.F32.size()) {
                        short s = b.getShort();
                        float v = Float.float16ToFloat(s);
                        bb.putFloat(i, v);
                    }
                    t = new FloatBufferTensor(bb.asFloatBuffer(), shape, true);
                } else {
                    sb = b.asShortBuffer().slice();
                    t = new Float16BufferTensor(name, sb, shape, true);
                }
                break;
            case BF16:
                sb = b.asShortBuffer().slice();
                t = new BFloat16BufferTensor(name, sb, shape, true);
                break;
            case Q4:
                FloatBufferTensor qb = (FloatBufferTensor) loader.load(name + ".qb");
                t = new Q4ByteBufferTensor(name, b.slice(), qb, shape, true);
                break;
            case I8:
                FloatBufferTensor qb1 = (FloatBufferTensor) loader.load(name + ".qb");
                t = new Q8ByteBufferTensor(name, b.slice(), qb1, shape, true);
                break;
            default:
                throw new IllegalArgumentException("Unsupported Tensor type: " + dType.name() + " for " + name);
        }

        return t;
    }

    @Override
    public AbstractTensor loadRows(String name, int rowOffset, int rowCount) {
        TensorInfo info = tensorInfoMap.get(name);
        if (info == null) {
            throw new NoSuchElementException(name + " not found in weights");
        }
        if (info.shape.length != 2) {
            throw new IllegalArgumentException("Row slicing only supported for 2D tensors: " + name);
        }
        if (info.dType == DType.Q4) {
            return loadQ4RowShard(name, rowOffset, rowCount, info);
        }
        if (info.dType == DType.I8) {
            throw new UnsupportedOperationException("Row slicing for I8 tensors is not supported: " + name);
        }

        int rows = Ints.checkedCast(info.shape[0]);
        int cols = Ints.checkedCast(info.shape[1]);
        if (rowOffset < 0 || rowCount < 0 || rowOffset + rowCount > rows) {
            throw new IllegalArgumentException("Invalid row range " + rowOffset + "," + rowCount + " for " + name);
        }

        int bytesPerRow = info.dType.size() * cols;
        long positionOffset = info.dataOffsets[0] + ((long) rowOffset * bytesPerRow);
        long positionLimit = positionOffset + ((long) rowCount * bytesPerRow);
        TensorShape shape = TensorShape.of(rowCount, cols);
        ByteBuffer b = bytes.duplicate()
                .order(ByteOrder.LITTLE_ENDIAN)
                .position(Ints.checkedCast(positionOffset))
                .limit(Ints.checkedCast(positionLimit));
        return loadTensorFromBuffer(name, info.dType, majorityDType, shape, b, parent.orElse(this));
    }

    @Override
    public AbstractTensor load(String name, TensorShardSpec shardSpec) {
        TensorInfo info = tensorInfoMap.get(name);
        if (info == null) {
            throw new NoSuchElementException(name + " not found in weights");
        }
        if (info.shape.length != 2) {
            throw new IllegalArgumentException("Tensor sharding only supported for 2D tensors: " + name);
        }
        return switch (shardSpec.axis()) {
            case ROWS -> loadRows(name, shardSpec.startInclusive(), shardSpec.length());
            case COLUMNS -> loadColumnShard(name, shardSpec, info);
        };
    }

    private AbstractTensor loadColumnShard(String name, TensorShardSpec shardSpec, TensorInfo info) {
        if (info.dType == DType.Q4) {
            return loadQ4ColumnShard(name, shardSpec, info);
        }
        if (info.dType == DType.I8) {
            throw new UnsupportedOperationException("Column sharding for I8 tensors is not supported: " + name);
        }
        try (AbstractTensor full = load(name)) {
            int rows = full.shape().first();
            int cols = full.shape().last();
            if (shardSpec.endExclusive() > cols) {
                throw new IllegalArgumentException("Invalid column range " + shardSpec.startInclusive() + ","
                        + shardSpec.endExclusive() + " for " + name);
            }
            AbstractTensor shard = allocateLike(info.dType, rows, shardSpec.length());
            for (int row = 0; row < rows; row++) {
                if (info.dType == DType.Q4 || info.dType == DType.I8) {
                    for (int col = 0; col < shardSpec.length(); col++) {
                        shard.set(full.get(row, shardSpec.startInclusive() + col), row, col);
                    }
                } else {
                    shard.copyFrom(full, full.getOffset(row, shardSpec.startInclusive()), shard.getOffset(row, 0),
                            shardSpec.length());
                }
            }
            return shard;
        }
    }

    private AbstractTensor loadQ4RowShard(String name, int rowOffset, int rowCount, TensorInfo info) {
        int rows = Ints.checkedCast(info.shape[0]);
        int cols = Ints.checkedCast(info.shape[1]);
        if (cols % Q4ByteBufferTensor.BLOCK_SIZE != 0) {
            throw new IllegalArgumentException("Q4 row sharding requires column count aligned to "
                    + Q4ByteBufferTensor.BLOCK_SIZE + ": " + name);
        }
        if (rowOffset < 0 || rowCount < 0 || rowOffset + rowCount > rows) {
            throw new IllegalArgumentException("Invalid row range " + rowOffset + "," + rowCount + " for " + name);
        }
        int bytesPerRow = cols / 2;
        long positionOffset = info.dataOffsets[0] + ((long) rowOffset * bytesPerRow);
        int byteLength = Ints.checkedCast((long) rowCount * bytesPerRow);
        ByteBuffer shardBytes = bytes.duplicate()
                .order(ByteOrder.LITTLE_ENDIAN)
                .position(Ints.checkedCast(positionOffset))
                .limit(Ints.checkedCast(positionOffset + byteLength))
                .slice()
                .order(ByteOrder.LITTLE_ENDIAN);
        FloatBufferTensor qBlocks = (FloatBufferTensor) parent.orElse(this).load(name + ".qb",
                new TensorShardSpec(TensorShardAxis.ROWS, rowOffset, rowOffset + rowCount));
        return new Q4ByteBufferTensor(name, shardBytes, qBlocks, TensorShape.of(rowCount, cols), true);
    }

    private AbstractTensor loadQ4ColumnShard(String name, TensorShardSpec shardSpec, TensorInfo info) {
        int rows = Ints.checkedCast(info.shape[0]);
        int cols = Ints.checkedCast(info.shape[1]);
        if (cols % Q4ByteBufferTensor.BLOCK_SIZE != 0) {
            throw new IllegalArgumentException("Q4 column sharding requires full column count aligned to "
                    + Q4ByteBufferTensor.BLOCK_SIZE + ": " + name);
        }
        if (shardSpec.startInclusive() % Q4ByteBufferTensor.BLOCK_SIZE != 0
                || shardSpec.endExclusive() % Q4ByteBufferTensor.BLOCK_SIZE != 0) {
            throw new IllegalArgumentException("Q4 column sharding requires shard range aligned to "
                    + Q4ByteBufferTensor.BLOCK_SIZE + ": " + name);
        }
        if (shardSpec.endExclusive() > cols) {
            throw new IllegalArgumentException("Invalid column range " + shardSpec.startInclusive() + ","
                    + shardSpec.endExclusive() + " for " + name);
        }
        int shardCols = shardSpec.length();
        int sourceBytesPerRow = cols / 2;
        int shardBytesPerRow = shardCols / 2;
        ByteBuffer shardBytes = ByteBuffer.allocateDirect(rows * shardBytesPerRow).order(ByteOrder.LITTLE_ENDIAN);
        for (int row = 0; row < rows; row++) {
            int sourceOffset = Ints.checkedCast(info.dataOffsets[0] + ((long) row * sourceBytesPerRow)
                    + (shardSpec.startInclusive() / 2));
            ByteBuffer source = bytes.duplicate()
                    .order(ByteOrder.LITTLE_ENDIAN)
                    .position(sourceOffset)
                    .limit(sourceOffset + shardBytesPerRow)
                    .slice();
            shardBytes.position(row * shardBytesPerRow);
            shardBytes.put(source);
        }
        shardBytes.clear();
        int blockStart = shardSpec.startInclusive() / Q4ByteBufferTensor.BLOCK_SIZE;
        int blockEnd = shardSpec.endExclusive() / Q4ByteBufferTensor.BLOCK_SIZE;
        FloatBufferTensor qBlocks = (FloatBufferTensor) parent.orElse(this).load(name + ".qb",
                new TensorShardSpec(TensorShardAxis.COLUMNS, blockStart, blockEnd));
        return new Q4ByteBufferTensor(name, shardBytes, qBlocks, TensorShape.of(rows, shardCols), true);
    }

    private static AbstractTensor allocateLike(DType dType, int rows, int cols) {
        TensorShape shape = TensorShape.of(rows, cols);
        return switch (dType) {
            case F32 -> new FloatBufferTensor(shape);
            case BF16 -> new BFloat16BufferTensor(shape);
            case F16 -> new Float16BufferTensor(shape);
            default -> throw new UnsupportedOperationException("Unsupported dtype for tensor shard: " + dType);
        };
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
