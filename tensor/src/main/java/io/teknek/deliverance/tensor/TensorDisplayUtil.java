package io.teknek.deliverance.tensor;

import java.util.Optional;

public class TensorDisplayUtil {

    public static String pretty2dDisplayAll(AbstractTensor t){
        return pretty2dDisplay(t, Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty());
    }
    public static String prettyHeader(AbstractTensor t){
        return  "###shape=" +
                t.shape +
                "\n" +
                "###uid=" +
                t.uid;
    }

    public static String pretty2dDisplay(AbstractTensor t, Optional<Integer> startRow, Optional<Integer> endRow,
                                   Optional<Integer> startColumn, Optional<Integer> endColumn, Optional<String> format) {
        String formatS = format.isPresent() ? format.get() : "%8.4f";
        StringBuilder sb = new StringBuilder();

        for (int i = startRow.orElse(0); i < endRow.orElse(t.shape.first()); i++) {
            for (int col = startColumn.orElse(0); col < endColumn.orElse(t.shape.dim(1)); col++) {
                //sb.append("[").append(i).append("]").append("[").append(col).append("]@").append((i * col) + col).append("=");
                sb.append("[").append(i).append("]").append("[").append(col).append("]").append("=");
                Optional<Integer> z=t.shape.safeOffset(i, col);
                if (z.isPresent()) {
                    sb.append(String.format(formatS, t.get(i, col)));
                    sb.append(" ");
                } else {
                    sb.append("out-of-bounds");
                }
                if (col == endColumn.orElse(t.shape.dim(1))-1){
                    sb.append("\n");
                }
            }
        }
        return sb.toString();
    }

    public static String pretty3dDisplayAll(AbstractTensor t) {
        return pretty3dDisplay(t, Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty());
    }

    public static String pretty3dDisplay(AbstractTensor t, Optional<Integer> startPlane, Optional<Integer> endPlane,
            Optional<Integer> startRow, Optional<Integer> endRow, Optional<Integer> startColumn,
            Optional<Integer> endColumn, Optional<String> format) {
        if (t.shape().dims() != 3) {
            throw new IllegalArgumentException("pretty3dDisplay requires a 3D tensor, got " + t.shape());
        }
        String formatS = format.orElse("%8.4f");
        StringBuilder sb = new StringBuilder();
        for (int plane = startPlane.orElse(0); plane < endPlane.orElse(t.shape().dim(0)); plane++) {
            sb.append("###plane[").append(plane).append("]\n");
            for (int row = startRow.orElse(0); row < endRow.orElse(t.shape().dim(1)); row++) {
                for (int col = startColumn.orElse(0); col < endColumn.orElse(t.shape().dim(2)); col++) {
                    sb.append("[").append(plane).append("][").append(row).append("][").append(col).append("]=");
                    if (plane >= 0 && plane < t.shape().dim(0)
                            && row >= 0 && row < t.shape().dim(1)
                            && col >= 0 && col < t.shape().dim(2)) {
                        sb.append(String.format(formatS, t.get(plane, row, col)));
                    } else {
                        sb.append("out-of-bounds");
                    }
                    sb.append(" ");
                }
                sb.append("\n");
            }
        }
        return sb.toString();
    }
}
