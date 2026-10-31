package io.teknek.deliverance.safetensors;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorInfo;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

class DefaultWeightLoaderMmapSplitTest {
    @Test
    void smallSplitContractUsesAbsoluteKeysAndRelativeContiguousParts() {
        Map<String, TensorInfo> infos = new LinkedHashMap<>();
        infos.put("model.ple.weight", new TensorInfo(
                DType.BF16, new long[]{10, 2}, new long[]{100, 140}));

        Map<List<Long>, List<String>> splits = DefaultWeightLoader.computeMmapSplits(infos, 200, 16);

        assertEquals(List.of("model.ple.weight-part-0"), splits.get(List.of(100L, 116L)));
        assertEquals(List.of("model.ple.weight-part-1"), splits.get(List.of(116L, 132L)));
        assertEquals(List.of("model.ple.weight-part-2"), splits.get(List.of(132L, 140L)));
        assertEquals(List.of(List.of(100L, 116L), List.of(116L, 132L), List.of(132L, 140L)),
                new ArrayList<>(splits.keySet()).stream().sorted((a, b) -> Long.compare(a.get(0), b.get(0))).toList());

        assertEquals(List.of(0L, 16L), List.of(infos.get("model.ple.weight-part-0").dataOffsets[0],
                infos.get("model.ple.weight-part-0").dataOffsets[1]));
        assertEquals(List.of(0L, 16L), List.of(infos.get("model.ple.weight-part-1").dataOffsets[0],
                infos.get("model.ple.weight-part-1").dataOffsets[1]));
        assertEquals(List.of(0L, 8L), List.of(infos.get("model.ple.weight-part-2").dataOffsets[0],
                infos.get("model.ple.weight-part-2").dataOffsets[1]));
    }

    @Test
    void localGemma4PleRowsMatchRawBf16Bytes() throws Exception {
        Path checkpoint = localGemma4Checkpoint();
        Assumptions.assumeTrue(checkpoint != null, "Set GEMMA4_CHECKPOINT or -Dgemma4.checkpoint");

        Map<String, TensorInfo> headerInfos = readHeader(checkpoint);
        String name = headerInfos.keySet().stream()
                .filter(key -> key.endsWith("embed_tokens_per_layer.weight"))
                .findFirst()
                .orElseThrow(() -> new IllegalArgumentException("Gemma4 PLE tensor not found"));
        TensorInfo info = headerInfos.get(name);
        Assumptions.assumeTrue(info.dType == DType.BF16, "PLE tensor is not BF16: " + info.dType);

        int rows = info.shape[0];
        int columns = info.shape[1];
        int bytesPerRow = columns * DType.BF16.size();
        int chunkRows = (Integer.MAX_VALUE - Integer.MAX_VALUE % bytesPerRow) / bytesPerRow;
        List<Integer> rowsToCheck = new ArrayList<>(List.of(0, 107, rows - 1));
        for (int row = chunkRows; row < rows; row += chunkRows) {
            rowsToCheck.add(row);
        }
        Path tensorFile = tensorFile(checkpoint, name);

        try (DefaultWeightLoader loader = new DefaultWeightLoader(checkpoint.toFile());
             RandomAccessFile file = new RandomAccessFile(tensorFile.toFile(), "r")) {
            for (int row : rowsToCheck) {
                byte[] raw = readRow(file.getChannel(), info, row, bytesPerRow);
                try (AbstractTensor loaded = loader.loadRows(name, row, 1)) {
                    for (int column = 0; column < columns; column++) {
                        int actual = Short.toUnsignedInt(ByteBuffer.wrap(raw, column * 2, 2)
                                .order(ByteOrder.LITTLE_ENDIAN).getShort());
                        int expected = Float.floatToRawIntBits(loaded.get(0, column)) >>> 16;
                        assertEquals(actual, expected, name + " row=" + row + " column=" + column);
                    }
                }
            }
        }
    }

    private static byte[] readRow(FileChannel channel, TensorInfo info, int row, int bytesPerRow) throws Exception {
        long position = 8L + headerLength(channel) + info.dataOffsets[0] + (long) row * bytesPerRow;
        ByteBuffer result = ByteBuffer.allocate(bytesPerRow);
        while (result.hasRemaining()) {
            int bytesRead = channel.read(result, position);
            if (bytesRead < 0) {
                throw new IllegalStateException("Unexpected end of safetensor payload");
            }
            position += bytesRead;
        }
        return result.array();
    }

    private static Map<String, TensorInfo> readHeader(Path checkpoint) throws Exception {
        Path tensorFile = tensorFile(checkpoint, "embed_tokens_per_layer.weight");
        try (RandomAccessFile file = new RandomAccessFile(tensorFile.toFile(), "r")) {
            FileChannel channel = file.getChannel();
            long headerLength = headerLength(channel);
            ByteBuffer header = channel.map(FileChannel.MapMode.READ_ONLY, 0, 8 + headerLength);
            return DefaultWeightLoader.readTensorInfoMap(header, java.util.Optional.empty());
        }
    }

    private static Path tensorFile(Path checkpoint, String tensorName) throws Exception {
        Path single = checkpoint.resolve(SafeTensorIndexPojo.SINGLE_MODEL_NAME);
        if (Files.isRegularFile(single)) {
            return single;
        }
        SafeTensorIndexPojo index = JsonUtils.om.readValue(
                checkpoint.resolve(SafeTensorIndexPojo.MODEL_INDEX_JSON).toFile(), SafeTensorIndexPojo.class);
        String fileName = index.getWeightFileMap().get(tensorName);
        if (fileName == null) {
            fileName = index.getWeightFileMap().entrySet().stream()
                    .filter(entry -> entry.getKey().endsWith("embed_tokens_per_layer.weight"))
                    .map(Map.Entry::getValue)
                    .findFirst()
                    .orElseThrow(() -> new IllegalArgumentException("PLE tensor file not found"));
        }
        return checkpoint.resolve(fileName);
    }

    private static long headerLength(FileChannel channel) throws Exception {
        ByteBuffer length = ByteBuffer.allocate(Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        channel.read(length, 0);
        return length.flip().getLong();
    }

    private static Path localGemma4Checkpoint() {
        String configured = System.getProperty("gemma4.checkpoint", System.getenv("GEMMA4_CHECKPOINT"));
        if (configured == null || configured.isBlank()) {
            return null;
        }
        Path path = Path.of(configured);
        if (Files.isRegularFile(path) && path.getFileName().toString().endsWith(".safetensors")) {
            return path.getParent();
        }
        return Files.isRegularFile(path.resolve(SafeTensorIndexPojo.SINGLE_MODEL_NAME))
                || Files.isRegularFile(path.resolve(SafeTensorIndexPojo.MODEL_INDEX_JSON)) ? path : null;
    }
}
