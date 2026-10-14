package io.teknek.deliverance.tensor2;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/** Lightweight opt-in CSV sink for provider timings during tensor tests. */
final class Tensor2TimingCsv {
    private static final Object LOCK = new Object();
    private static final String HEADER = "operation,provider,kernel,threading,a_dtype,b_dtype,result_dtype,"
            + "a_rows,a_columns,b_rows,b_columns,result_rows,result_columns,a_row_offset,a_column_offset,"
            + "b_column_offset,column_length,result_row_offset,b_row_offset,row_chunk_size,elapsed_ns,mean_ms,gflops\n";

    private Tensor2TimingCsv() {
    }

    static void record(BatchDotProduct operation, TensorProviderKind provider, long elapsedNanos) {
        if (!Boolean.getBoolean("deliverance.tensor2.timing.csv")) {
            return;
        }
        TensorRef a = operation.a();
        TensorRef b = operation.b();
        TensorRef result = operation.result();
        double meanMs = elapsedNanos / 1_000_000.0;
        double gflops = 2.0 * result.shape().first() * operation.columnLength() * operation.rowChunkSize()
                / (elapsedNanos * 1_000_000.0);
        String row = String.join(",",
                "batchDotProduct", provider.name(), operation.kernel().name(), operation.threading().name(),
                a.dType().name(), b.dType().name(), result.dType().name(),
                Long.toString(a.shape().first()), Long.toString(a.shape().last()),
                Long.toString(b.shape().first()), Long.toString(b.shape().last()),
                Long.toString(result.shape().first()), Long.toString(result.shape().last()),
                Integer.toString(operation.aRowOffset()), Integer.toString(operation.aColumnOffset()),
                Integer.toString(operation.bColumnOffset()), Integer.toString(operation.columnLength()),
                Integer.toString(operation.resultRowOffset()), Integer.toString(operation.bRowOffset()),
                Integer.toString(operation.rowChunkSize()), Long.toString(elapsedNanos),
                String.format(java.util.Locale.ROOT, "%.6f", meanMs),
                String.format(java.util.Locale.ROOT, "%.6f", gflops)) + "\n";
        synchronized (LOCK) {
            Path path = Path.of(System.getProperty("deliverance.tensor2.timing.csv.path",
                    "target/tensor2-fuzzer-timings.csv"));
            try {
                if (Files.notExists(path)) {
                    Files.createDirectories(path.toAbsolutePath().getParent());
                    Files.writeString(path, HEADER, StandardOpenOption.CREATE, StandardOpenOption.WRITE);
                }
                Files.writeString(path, row, StandardOpenOption.CREATE, StandardOpenOption.APPEND);
            } catch (IOException e) {
                throw new IllegalStateException("Unable to write tensor2 timing CSV: " + path, e);
            }
        }
    }
}
