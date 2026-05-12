package io.teknek.deliverance.safetensors;

import java.nio.file.Path;

/**
 * Writes local model inspection/comparison reports to JSON files for checkpoint debugging.
 */
public final class ModelInspectionMain {
    private ModelInspectionMain() {
    }

    public static void main(String[] args) {
        //if (args.length < 1) {
        //    throw new IllegalArgumentException(usage());
        //}
        args = new String[] { "compare", "google", "gemma-2-2b-it", "tjake", "gemma-2-2b-it-JQ4", "/Users/edward.capriolo/gemma_comp.json"};
        ModelInspectionTool tool = new ModelInspectionTool();
        switch (args[0]) {
            case "inspect" -> {
                if (args.length != 4) {
                    throw new IllegalArgumentException(usage());
                }
                var report = tool.inspectCachedModel(args[1], args[2]);
                tool.writeJsonReport(Path.of(args[3]), report);
            }
            case "compare" -> {
                if (args.length != 6) {
                    throw new IllegalArgumentException(usage());
                }
                var report = tool.compareCachedModels(args[1], args[2], args[3], args[4]);
                tool.writeJsonReport(Path.of(args[5]), report);
            }
            default -> throw new IllegalArgumentException(usage());
        }
    }

    private static String usage() {
        return "usage: ModelInspectionMain inspect <owner> <model> <output.json> | compare <leftOwner> <leftModel> <rightOwner> <rightModel> <output.json>";
    }
}
