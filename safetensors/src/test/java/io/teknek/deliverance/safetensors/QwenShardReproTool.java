package io.teknek.deliverance.safetensors;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.AbstractTensor;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

/** Manual local repro tool for a single downloaded Qwen3 shard. Not a unit test. */
public final class QwenShardReproTool {
    private QwenShardReproTool() {
    }

    public static void main(String[] args) throws Exception {
        Path shardDir = Path.of(args.length > 0 ? args[0] : "/ai-code/qwen3-32b-shard-repro");
        Path outputDir = Path.of(args.length > 1 ? args[1] : "/tmp/qwen3-32b-shard-repro-q4");
        String tensorName = args.length > 2 ? args[2] : "model.layers.12.mlp.down_proj.weight";
        System.out.println("repro shard_dir=" + shardDir);
        System.out.println("repro output_dir=" + outputDir);
        System.out.println("repro tensor=" + tensorName);
        try (SafetensorsShardWeightLoader loader = new SafetensorsShardWeightLoader(shardDir.resolve("model-00004-of-00017.safetensors"))) {
            System.out.println("tensor_count=" + loader.tensorInfoMap().size());
            System.out.println("info=" + loader.tensorInfoMap().get(tensorName));
            long before = usedMemory();
            try (AbstractTensor tensor = loader.load(tensorName)) {
                System.out.println("loaded dtype=" + tensor.dType() + " shape=" + tensor.shape() + " size=" + tensor.size());
                System.out.println("used_memory_after_load=" + usedMemory() + " delta=" + (usedMemory() - before));
                Path miniSource = outputDir.resolveSibling(outputDir.getFileName() + "-source");
                Files.createDirectories(miniSource);
                Files.copy(shardDir.resolve("config.json"), miniSource.resolve("config.json"), java.nio.file.StandardCopyOption.REPLACE_EXISTING);
                Path shardLink = miniSource.resolve("model-00004-of-00017.safetensors");
                Files.deleteIfExists(shardLink);
                try {
                    Files.createSymbolicLink(shardLink, shardDir.resolve("model-00004-of-00017.safetensors"));
                } catch (UnsupportedOperationException | java.io.IOException e) {
                    Files.copy(shardDir.resolve("model-00004-of-00017.safetensors"), shardLink,
                            java.nio.file.StandardCopyOption.REPLACE_EXISTING);
                }
                JsonUtils.om.writeValue(miniSource.resolve(SafeTensorIndexPojo.MODEL_INDEX_JSON).toFile(),
                        new SafeTensorIndexPojo(Map.of(), Map.of(tensorName, "model-00004-of-00017.safetensors")));
                new ModelQuantizer(64L * 1024L * 1024L, ModelQuantizer.ReadMode.SHARD_WEIGHT_LOADER)
                        .quantizeModelDirectory(miniSource, outputDir, DType.Q4, name -> name.equals(tensorName));
            }
            System.out.println("used_memory_after_close=" + usedMemory());
        }
        try (DefaultWeightLoader output = new DefaultWeightLoader(outputDir.toFile())) {
            System.out.println("output_info=" + output.tensorInfoMap().get(tensorName));
            System.out.println("output_qb_info=" + output.tensorInfoMap().get(tensorName + ".qb"));
            System.out.println("output_has_qb=" + output.isWeightPresent(tensorName + ".qb"));
        }
    }

    private static long usedMemory() {
        Runtime runtime = Runtime.getRuntime();
        return runtime.totalMemory() - runtime.freeMemory();
    }
}
