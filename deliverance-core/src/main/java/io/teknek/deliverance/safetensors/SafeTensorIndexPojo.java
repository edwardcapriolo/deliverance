package io.teknek.deliverance.safetensors;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.google.common.collect.ImmutableMap;

import java.util.Map;


public class SafeTensorIndexPojo {

    public static final String MODEL_INDEX_JSON =  "model.safetensors.index.json";
    public static final String SINGLE_MODEL_NAME = "model.safetensors";
    private final Map<String, String> metadata;
    private final Map<String, String> weightFileMap;

    @JsonCreator
    public SafeTensorIndexPojo(@JsonProperty("metadata") Map<String, String> metadata,
                               @JsonProperty("weight_map") Map<String, String> weightFileMap) {
        this.metadata = ImmutableMap.copyOf(metadata);
        this.weightFileMap = ImmutableMap.copyOf(weightFileMap);
    }


    public Map<String, String> getMetadata() {
        return metadata;
    }

    public Map<String, String> getWeightFileMap() {
        return weightFileMap;
    }


/*
    static void loadWeights(SafeTensorIndex index, Path modelRoot) throws IOException {
        for (Map.Entry<String, String> e : index.weightFileMap.entrySet()) {
            // Only load the file if it's not already loaded
            if (!index.fileMap.containsKey(e.getValue())) {
                RandomAccessFile raf = new RandomAccessFile(Paths.get(modelRoot.toString(), e.getValue()).toFile(), "r");
                index.fileMap.put(e.getValue(), raf);

                // Read the first 1MB of the file to get the TensorInfo
                ByteBuffer header = raf.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, Math.min(1 << 20, raf.length()));

                Map<String, String> metadata = new HashMap<>();
                Map<String, TensorInfo> tensorInfoMap = SafeTensorSupport.readTensorInfoMap(header, Optional.of(metadata));
                index.allTensorInfoMap.putAll(tensorInfoMap);
                int endOfHeaderPosition = header.position();

                Map<List<Long>, List<String>> splits = index.computeMmapSplits(tensorInfoMap, raf.length());
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

                    Weights mmapWeights = new Weights(metadata, mmapTensorInfoMap, buf, Optional.of(index));
                    for (String tensor : tensors) {
                        index.weightMap.put(tensor, mmapWeights);
                    }
                }
            }
        }
    }

 */
    /*
    public static SafeTensorIndex loadWithWeights(Path modelRoot) {
        try {
            File indexFile = Paths.get(modelRoot.toString(), MODEL_INDEX_JSON).toFile();

            SafeTensorIndex index = om.readValue(indexFile, SafeTensorIndex.class);
            loadWeights(index, modelRoot);

            return index;
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }*/


/*
    public SafeTensorIndex(Path modelRoot){
        File indexFile = Paths.get(modelRoot.toString(), MODEL_INDEX_JSON).toFile();

        try {
            SafeTensorIndex index = om.readValue(indexFile, SafeTensorIndex.class);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

    }*/




    /*
    public void load(){

    }*/
}
