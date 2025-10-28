package io.teknek.deliverance.safetensors;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.type.MapType;
import com.google.common.collect.ImmutableMap;
import com.google.common.primitives.Ints;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.model.DistributedContext;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorInfo;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.UncheckedIOException;
import java.util.LinkedHashMap;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

import static io.teknek.deliverance.JsonUtils.om;
import static io.teknek.deliverance.safetensors.Weights.findDType;

public class DefaultWeightLoader implements WeightLoader {

    public static final Logger LOGGER = LoggerFactory.getLogger(DefaultWeightLoader.class);

    private static final MapType metadataTypeReference = om.getTypeFactory().constructMapType(Map.class, String.class, String.class);

    private final SafeTensorIndexPojo index;
    private final ConcurrentHashMap<String, RandomAccessFile> fileMap = new ConcurrentHashMap<>();
    private final Map<String, TensorInfo> allTensorInfoMap = new ConcurrentHashMap<>();
    private final Map<String, Weights> weightMap = new ConcurrentHashMap<>();
    private final Path modelRoot;
    private final DType majorityDType;


    public DefaultWeightLoader(File baseDir){
        this.modelRoot = Paths.get(baseDir.toURI());
        Path singleFile = Paths.get(baseDir.getAbsolutePath(), SafeTensorIndexPojo.SINGLE_MODEL_NAME);
        if (Files.exists(Paths.get(baseDir.getAbsolutePath(), SafeTensorIndexPojo.MODEL_INDEX_JSON))){
            throw new IllegalArgumentException("Not implemented");
        } else if (Files.exists(singleFile)){
            index = new SafeTensorIndexPojo(Collections.emptyMap(), Map.of("model-file", singleFile.toFile().getName()));
        } else {
            throw new IllegalArgumentException("weights not found");
        }

        try {
            loadWeights();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        this.majorityDType = findDType(allTensorInfoMap);
        System.out.println("Majority DType "+ this.majorityDType);
    }

    public void loadWeights() throws IOException {
        for (Map.Entry<String, String> e : index.getWeightFileMap().entrySet()) {
            if (!fileMap.containsKey(e.getValue())) {
                RandomAccessFile raf = new RandomAccessFile(Paths.get(modelRoot.toString(), e.getValue()).toFile(), "r");
                fileMap.put(e.getValue(), raf);
                ByteBuffer header = raf.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, Math.min(1 << 20, raf.length()));

                Map<String, String> metadata = new HashMap<>();
                Map<String, TensorInfo> tensorInfoMap = readTensorInfoMap(header, Optional.of(metadata));
                allTensorInfoMap.putAll(tensorInfoMap);
                int endOfHeaderPosition = header.position();

                Map<List<Long>, List<String>> splits = computeMmapSplits(tensorInfoMap, raf.length());
                for (Map.Entry<List<Long>, List<String>> split : splits.entrySet()) {
                    long offset = split.getKey().get(0);
                    long length = split.getKey().get(1);
                    List<String> tensors = split.getValue();
                    int lengthInt = Ints.checkedCast(length - offset);
                    ByteBuffer buf = raf.getChannel().map(FileChannel.MapMode.READ_ONLY, endOfHeaderPosition + offset, lengthInt);
                    Map<String, TensorInfo> mmapTensorInfoMap = tensorInfoMap.entrySet()
                            .stream()
                            .filter(x -> tensors.contains(x.getKey()))
                            .collect(ImmutableMap.toImmutableMap(Map.Entry::getKey, Map.Entry::getValue));
                    Weights mmapWeights = new Weights(metadata, mmapTensorInfoMap, buf, Optional.of(this));
                    for (String tensor : tensors) {
                        weightMap.put(tensor, mmapWeights);
                    }
                }
            }
        }
    }

    public static Map<String, TensorInfo> readTensorInfoMap(ByteBuffer buf, Optional<Map<String, String>> saveMetadata) {
        final long MAX_HEADER_LENGTH = 1024 * 1024 * 1024; // 1 GB
        buf = buf.order(ByteOrder.LITTLE_ENDIAN);
        long headerLength = buf.getLong();
        if (headerLength < 0) {
            throw new IllegalArgumentException("Header length cannot be negative: " + headerLength);
        }
        if (headerLength > MAX_HEADER_LENGTH) {
            throw new IllegalArgumentException(
                    String.format("Header length %d exceeds the maximum allowed length %d.", headerLength, MAX_HEADER_LENGTH)
            );
        }
        byte[] header = new byte[Ints.checkedCast(headerLength)];
        buf.get(header);

        try {
            JsonNode rootNode = om.readTree(header);
            Iterator<Map.Entry<String, JsonNode>> fields = rootNode.fields();
            Map<String, TensorInfo> tensorInfoMap = new HashMap<>();
            Map<String, String> metadata = Collections.emptyMap();

            while (fields.hasNext()) {
                Map.Entry<String, JsonNode> field = fields.next();
                if (field.getKey().equalsIgnoreCase("__metadata__")) {
                    metadata = om.treeToValue(field.getValue(), metadataTypeReference);
                } else {
                    TensorInfo tensorInfo = om.treeToValue(field.getValue(), TensorInfo.class);
                    tensorInfoMap.put(field.getKey(), tensorInfo);
                }
            }
            Map<String, TensorInfo> sortedMap = tensorInfoMap.entrySet()
                    .stream()
                    .sorted(Map.Entry.comparingByValue())
                    .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue, (e1, e2) -> e1, LinkedHashMap::new));

            final Map<String, String> finalMetadata = metadata;
            saveMetadata.ifPresent(m -> m.putAll(finalMetadata));

            return sortedMap;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private Map<List<Long>, List<String>> computeMmapSplits(Map<String, TensorInfo> tensorInfoMap, long fileLength) {
        Map<List<Long>, List<String>> splits = new HashMap<>();
        long lastSplitOffset = 0;
        int tensorsInFile = tensorInfoMap.size();
        int tensorsSplit = 0;
        List<String> tensors = new ArrayList<>();

        Iterator<Map.Entry<String, TensorInfo>> it = new ArrayList<>(tensorInfoMap.entrySet()).iterator();
        Map.Entry<String, TensorInfo> next = null;
        while (tensorsSplit < tensorsInFile && (it.hasNext() || next != null)) {
            tensors.clear();
            long limit = lastSplitOffset + Integer.MAX_VALUE;
            long startOffset = fileLength;
            long endOffset = 0;

            while (it.hasNext() || next != null) {
                //this looks suspicious
                next = next == null ? it.next() : next;
                TensorInfo info = next.getValue();
                LOGGER.debug("Tensor {} {} {} limit {}", next.getKey(), info.dataOffsets[0], info.dataOffsets[1], limit);
                if (info.dataOffsets[1] < limit) {
                    tensors.add(next.getKey());
                    tensorsSplit++;

                    if (info.dataOffsets[1] > endOffset) endOffset = info.dataOffsets[1];
                    if (info.dataOffsets[0] < startOffset) startOffset = info.dataOffsets[0];
                    info.dataOffsets[0] -= lastSplitOffset;
                    info.dataOffsets[1] -= lastSplitOffset;

                    LOGGER.debug("Adding tensor {} to split {}-{}", next.getKey(), info.dataOffsets[0], info.dataOffsets[1]);
                    next = null;
                } else {
                    // Split large tensors up (they will be reassembled in the Weights class)
                    if (tensors.isEmpty()) {
                        int bytesPerColumn = info.dType.size() * info.shape[1];

                        // This tensor is too large to fit in a single split
                        // We'll split it up into smaller chunks
                        if (info.dataOffsets[1] > endOffset) endOffset = info.dataOffsets[1];
                        if (info.dataOffsets[0] < startOffset) startOffset = info.dataOffsets[0];

                        // Adjust the offset to be relative to the start of the split
                        info.dataOffsets[0] -= lastSplitOffset;
                        info.dataOffsets[1] -= lastSplitOffset;

                        long offset = info.dataOffsets[0];
                        long length = info.dataOffsets[1] - offset;

                        // Chunk size needs to be a multiple of the column size
                        long chunkSize = Integer.MAX_VALUE - (Integer.MAX_VALUE % bytesPerColumn);
                        long offsetAdded = 0;
                        int chunk = 0;
                        boolean added = false;
                        while (length > 0) {
                            long chunkEnd = Math.min(offset + chunkSize, endOffset);
                            String chunkName = next.getKey() + "-part-" + chunk++;
                            LOGGER.debug(
                                    "Adding chunk {} to split {}-{} {}",
                                    chunkName,
                                    offset,
                                    chunkEnd,
                                    Ints.checkedCast(chunkEnd - offset)
                            );
                            splits.put(List.of(offset, chunkEnd), List.of(chunkName));

                            // Add TensorInfo for the chunk
                            if (info.shape.length != 2){
                                throw new RuntimeException("Only 2D tensors supported");
                            }
                            int numRowsInChunk = Ints.checkedCast((chunkEnd - offset) / bytesPerColumn);

                            // This tensorInfo is relative to the split which we know is at least the mmap limit
                            // We track the offsetAdded so we can make the offset relative to the current split
                            TensorInfo chunkInfo = new TensorInfo(
                                    info.dType,
                                    new long[] { numRowsInChunk, info.shape[1] },
                                    new long[] { offset - offsetAdded, chunkEnd - offsetAdded }
                            );
                            tensorInfoMap.put(chunkName, chunkInfo);
                            added = true;
                            offsetAdded += chunkEnd - offset;

                            offset = chunkEnd;
                            length -= chunkSize;
                        }

                        if (added) {
                            tensorsSplit++;
                            next = null;
                        }
                    }

                    break;
                }
            }

            if (tensorsSplit <= 0){
                throw new RuntimeException(" no tensors to split");
            }
            LOGGER.debug("Adding split {}-{} with {} tensors of {}", startOffset, endOffset, tensors.size(), tensorsSplit);
            if (!tensors.isEmpty()) {
                splits.put(List.of(startOffset, endOffset), new ArrayList<>(tensors));
            }
            if (endOffset > lastSplitOffset) lastSplitOffset = endOffset;
        }

        assert tensorsInFile == tensorsSplit : "Not all tensors were split: " + tensorsSplit + " != " + tensorsInFile;
        return splits;
    }


    @Override
    public Map<String, String> metadata() {
        return this.index.getMetadata();
    }

    @Override
    public Map<String, TensorInfo> tensorInfoMap() {
        return this.allTensorInfoMap;
    }

    @Override
    public AbstractTensor load(String name, DistributedContext dctx, boolean sparseRows, boolean sparseColumns) {
        Weights w = this.weightMap.get(name);
        return w.load(name);
    }

    @Override
    public DType getModelDType() {

        return this.majorityDType;
    }

    @Override
    public void close() {
        for (Map.Entry<?, RandomAccessFile> entry : fileMap.entrySet()){
            try {
                entry.getValue().close();
            } catch (IOException e){
                LOGGER.warn("issue during close", e);
            }

        }

        //this.index = null;
        this.allTensorInfoMap.clear();
        this.weightMap.clear();
        this.fileMap.clear();
    }

    @Override
    public String toString() {
        return "DefaultWeightLoader{" +
                "modelRoot=" + modelRoot +
                ", index=" + index +
                ", weightMap=" + weightMap +
                ", allTensorInfoMap=" + allTensorInfoMap +
                '}';
    }
}
