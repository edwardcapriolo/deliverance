package io.teknek.reckoning.safetensors;

import java.nio.file.Path;

/**
 * Writes local model inspection/comparison reports to JSON files for checkpoint debugging.
 */
public final class ModelInspectionMain {
    private ModelInspectionMain() {
    }

    public static void main(String[] args) {
        if (args.length < 1) {
            throw new IllegalArgumentException(usage());
        }
        ModelInspectionTool tool = new ModelInspectionTool();
        switch (args[0]) {
            case "inspect" -> {
                if (args.length != 4 && args.length != 5) {
                    throw new IllegalArgumentException(usage());
                }
                var report = tool.inspectCachedModel(args[1], args[2]);
                Path output = Path.of(args[3]);
                if (args.length == 5 && "markdown".equals(args[4])) {
                    tool.writeMarkdownReport(output, report);
                } else {
                    tool.writeJsonReport(output, report);
                }
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
        return "usage: ModelInspectionMain inspect <owner> <model> <output> [markdown] | compare <leftOwner> <leftModel> <rightOwner> <rightModel> <output.json>";
    }
}
