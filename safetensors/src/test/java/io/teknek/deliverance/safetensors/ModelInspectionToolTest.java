package io.teknek.deliverance.safetensors;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorInfo;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ModelInspectionToolTest {
    @Test
    void comparisonSummarizesDtypeChanges() {
        ModelInspectionTool.TensorInspection left = new ModelInspectionTool.TensorInspection(
                "layer.weight", DType.F32, new int[]{2, 2}, 4, 16, false, false, true, true,
                "selected and eligible for Q4 quantization"
        );
        ModelInspectionTool.TensorInspection right = new ModelInspectionTool.TensorInspection(
                "layer.weight", DType.Q4, new int[]{2, 2}, 4, 8, false, false, true, false,
                "selected by filter but not quantized"
        );

        String summary = invokeSummary(left, right);
        assertEquals("dtype changed from F32 to Q4", summary);
    }

    @Test
    void comparisonClassifiesExpectedQbSidecar() {
        ModelInspectionTool.TensorInspection right = new ModelInspectionTool.TensorInspection(
                "layer.weight.qb", DType.F32, new int[]{2, 1}, 2, 8, false, false, false, false,
                "split tensor chunk"
        );
        String summary = invokeSummary(null, right);
        assertEquals("only present on right (expected q4/q8 sidecar)", summary);
    }

    @Test
    void tensorInfoEstimatedBytesAssumptionsMatch() {
        TensorInfo info = new TensorInfo(DType.F32, new long[]{3, 4}, new long[]{0, 48});
        assertEquals(12L, invokeElementCount(info.shape));
        assertEquals(48L, invokeEstimatedBytes(info));
    }

    @Test
    void quantizationNoteExplainsNonTwoDimensionalSkip() {
        TensorInfo info = new TensorInfo(DType.F32, new long[]{2, 2, 8}, new long[]{0, 128});
        String note = invokeQuantizationNote("conv.weight", info, true, false, false, false);
        assertTrue(note.contains("current quantizer only handles 2D tensors"));
    }

    private static String invokeSummary(ModelInspectionTool.TensorInspection left, ModelInspectionTool.TensorInspection right) {
        try {
            var m = ModelInspectionTool.class.getDeclaredMethod("summarizeChange",
                    ModelInspectionTool.TensorInspection.class,
                    ModelInspectionTool.TensorInspection.class,
                    boolean.class,
                    boolean.class);
            m.setAccessible(true);
            return (String) m.invoke(null, left, right, true, false);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static long invokeElementCount(int[] shape) {
        try {
            var m = ModelInspectionTool.class.getDeclaredMethod("elementCount", int[].class);
            m.setAccessible(true);
            return (long) m.invoke(null, (Object) shape);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static long invokeEstimatedBytes(TensorInfo info) {
        try {
            var m = ModelInspectionTool.class.getDeclaredMethod("estimatedBytes", TensorInfo.class);
            m.setAccessible(true);
            return (long) m.invoke(null, info);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static String invokeQuantizationNote(String name, TensorInfo info, boolean selected, boolean quantizable,
                                                 boolean logicalParent, boolean splitPart) {
        try {
            var m = ModelInspectionTool.class.getDeclaredMethod("quantizationNote",
                    String.class, TensorInfo.class, boolean.class, boolean.class, boolean.class, boolean.class);
            m.setAccessible(true);
            return (String) m.invoke(null, name, info, selected, quantizable, logicalParent, splitPart);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
