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
}
