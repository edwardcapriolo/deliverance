package io.teknek.deliverance.safetensors;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.tensor.TensorInfo;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;

/**
 * Produces file-based inspection reports for local model directories so original and quantized
 * checkpoints can be compared without ad hoc debugging or loading them into generation code.
 */
public class ModelInspectionTool {
    public record TensorInspection(
            String name,
            DType dType,
            int[] shape,
            long elementCount,
            long estimatedBytes,
            boolean logicalSplitParent,
            boolean splitPart,
            boolean quantizerSelected,
            boolean quantizableToQ4,
            String quantizationNote
    ) {
    }

    public record ModelInspectionReport(
            String source,
            Path modelDirectory,
            Map<String, String> metadata,
            int tensorCount,
            Map<String, Integer> dtypeCounts,
            int logicalSplitParentCount,
            int splitPartCount,
            int quantizerSelectedCount,
            int quantizableToQ4Count,
            List<TensorInspection> tensors
    ) {
    }

    public record TensorComparison(
            String name,
            boolean leftPresent,
            boolean rightPresent,
            DType leftDType,
            DType rightDType,
            int[] leftShape,
            int[] rightShape,
            boolean sameShape,
            boolean sameDType,
            String changeSummary
    ) {
    }

    public record ModelComparisonReport(
            String leftSource,
            String rightSource,
            Path leftDirectory,
            Path rightDirectory,
            int comparedTensorCount,
            int missingOnRightCount,
            int dtypeChangedCount,
            int shapeChangedCount,
            List<TensorComparison> tensors
    ) {
    }

    public ModelInspectionReport inspectCachedModel(String owner, String model) {
        return inspectModelDirectory(owner + "/" + model, new ModelFetcher(owner, model).pathForModel());
    }

    public ModelInspectionReport inspectModelDirectory(String source, Path modelDirectory) {
        if (!Files.isDirectory(modelDirectory)) {
            throw new IllegalArgumentException("Model directory not found: " + modelDirectory);
        }

        ModelQuantizer quantizer = new ModelQuantizer();
        try (DefaultWeightLoader loader = new DefaultWeightLoader(modelDirectory.toFile())) {
            ArrayList<TensorInspection> tensors = new ArrayList<>();
            LinkedHashMap<String, Integer> dtypeCounts = new LinkedHashMap<>();
            int logicalParents = 0;
            int splitParts = 0;
            int quantizerSelected = 0;
            int quantizableToQ4 = 0;

            for (Map.Entry<String, TensorInfo> entry : loader.tensorInfoMap().entrySet()) {
                String name = entry.getKey();
                TensorInfo info = entry.getValue();
                boolean logicalParent = quantizer.isLogicalSplitTensor(name, loader);
                boolean splitPart = name.contains("-part-");
                boolean selected = ModelQuantizer.DEFAULT_Q4_TENSOR_FILTER.test(name);
                boolean quantizable = quantizer.canQuantize(info, DType.Q4);
                String note = quantizationNote(name, info, selected, quantizable, logicalParent, splitPart);

                if (logicalParent) logicalParents++;
                if (splitPart) splitParts++;
                if (selected) quantizerSelected++;
                if (quantizable) quantizableToQ4++;
                dtypeCounts.merge(info.dType.name(), 1, Integer::sum);

                tensors.add(new TensorInspection(
                        name,
                        info.dType,
                        info.shape.clone(),
                        elementCount(info.shape),
                        estimatedBytes(info),
                        logicalParent,
                        splitPart,
                        selected,
                        quantizable,
                        note
                ));
            }

            return new ModelInspectionReport(
                    source,
                    modelDirectory,
                    loader.metadata(),
                    tensors.size(),
                    dtypeCounts,
                    logicalParents,
                    splitParts,
                    quantizerSelected,
                    quantizableToQ4,
                    List.copyOf(tensors)
            );
        }
    }

    public ModelComparisonReport compareCachedModels(String leftOwner, String leftModel, String rightOwner, String rightModel) {
        return compareModelDirectories(
                leftOwner + "/" + leftModel,
                new ModelFetcher(leftOwner, leftModel).pathForModel(),
                rightOwner + "/" + rightModel,
                new ModelFetcher(rightOwner, rightModel).pathForModel()
        );
    }

    public ModelComparisonReport compareModelDirectories(String leftSource, Path leftDirectory, String rightSource, Path rightDirectory) {
        ModelInspectionReport left = inspectModelDirectory(leftSource, leftDirectory);
        ModelInspectionReport right = inspectModelDirectory(rightSource, rightDirectory);

        Map<String, TensorInspection> leftMap = indexByName(left.tensors());
        Map<String, TensorInspection> rightMap = indexByName(right.tensors());
        TreeSet<String> allNames = new TreeSet<>();
        allNames.addAll(leftMap.keySet());
        allNames.addAll(rightMap.keySet());

        ArrayList<TensorComparison> comparisons = new ArrayList<>();
        int missingOnRight = 0;
        int dtypeChanged = 0;
        int shapeChanged = 0;

        for (String name : allNames) {
            TensorInspection l = leftMap.get(name);
            TensorInspection r = rightMap.get(name);
            boolean leftPresent = l != null;
            boolean rightPresent = r != null;
            boolean sameShape = leftPresent && rightPresent && java.util.Arrays.equals(l.shape(), r.shape());
            boolean sameDType = leftPresent && rightPresent && l.dType() == r.dType();
            String summary = summarizeChange(l, r, sameShape, sameDType);
            if (leftPresent && !rightPresent) missingOnRight++;
            if (leftPresent && rightPresent && !sameDType) dtypeChanged++;
            if (leftPresent && rightPresent && !sameShape) shapeChanged++;

            comparisons.add(new TensorComparison(
                    name,
                    leftPresent,
                    rightPresent,
                    leftPresent ? l.dType() : null,
                    rightPresent ? r.dType() : null,
                    leftPresent ? l.shape() : null,
                    rightPresent ? r.shape() : null,
                    sameShape,
                    sameDType,
                    summary
            ));
        }

        return new ModelComparisonReport(
                leftSource,
                rightSource,
                leftDirectory,
                rightDirectory,
                comparisons.size(),
                missingOnRight,
                dtypeChanged,
                shapeChanged,
                List.copyOf(comparisons)
        );
    }

    public void writeJsonReport(Path outputFile, Object report) {
        try {
            if (outputFile.getParent() != null) {
                Files.createDirectories(outputFile.getParent());
            }
            JsonUtils.om.writerWithDefaultPrettyPrinter().writeValue(outputFile.toFile(), report);
        } catch (IOException e) {
            throw new UncheckedIOException("Unable to write report to " + outputFile, e);
        }
    }

    private static Map<String, TensorInspection> indexByName(List<TensorInspection> tensors) {
        LinkedHashMap<String, TensorInspection> map = new LinkedHashMap<>();
        for (TensorInspection tensor : tensors) {
            map.put(tensor.name(), tensor);
        }
        return map;
    }

    private static long elementCount(int[] shape) {
        if (shape.length == 0) {
            return 1;
        }
        long count = 1;
        for (int dim : shape) {
            count *= dim;
        }
        return count;
    }

    private static long estimatedBytes(TensorInfo info) {
        long elements = elementCount(info.shape);
        return elements * info.dType.size();
    }

    private static String quantizationNote(String name, TensorInfo info, boolean selected, boolean quantizable,
                                           boolean logicalParent, boolean splitPart) {
        if (logicalParent) {
            return "logical split parent; concrete -part-* tensors carry the real data";
        }
        if (splitPart) {
            return "split tensor chunk";
        }
        if (!selected) {
            if (!name.endsWith(".weight")) {
                return "not selected by default filter because it is not a weight tensor";
            }
            if (name.endsWith("embed_tokens.weight") || name.endsWith("embed_tokens_per_layer.weight") || name.endsWith("lm_head.weight")) {
                return "kept dense by default filter because it is an embedding or lm_head";
            }
            return "skipped by default filter";
        }
        if (!quantizable) {
            if (info.shape.length != 2) {
                return "selected by filter but kept dense because current quantizer only handles 2D tensors";
            }
            return "selected by filter but not quantized";
        }
        return "selected and eligible for Q4 quantization";
    }

    private static String summarizeChange(TensorInspection left, TensorInspection right, boolean sameShape, boolean sameDType) {
        if (left == null) {
            if (right != null && right.name().endsWith(".qb")) {
                return "only present on right (expected q4/q8 sidecar)";
            }
            return "only present on right (unexpected)";
        }
        if (right == null) {
            return "missing on right";
        }
        if (!sameDType && sameShape) {
            return "dtype changed from " + left.dType() + " to " + right.dType();
        }
        if (!sameShape && sameDType) {
            return "shape changed while dtype stayed " + left.dType();
        }
        if (!sameShape) {
            return "dtype and shape changed";
        }
        return "unchanged";
    }
}
