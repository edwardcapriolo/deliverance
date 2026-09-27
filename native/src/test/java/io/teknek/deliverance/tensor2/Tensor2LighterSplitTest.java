package io.teknek.deliverance.tensor2;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class Tensor2LighterSplitTest {
    @Test
    void splitMaterializesEqualChunksAlongLastDimension() {
        Lighter lighter = new Lighter(new MetricRegistry());
        try (TensorRef source = lighter.allocate(DType.F32, TensorShape.of(2, 12))) {
            for (int row = 0; row < 2; row++) {
                for (int column = 0; column < 12; column++) {
                    source.set(row * 100.0f + column, row, column);
                }
            }
            TensorRef[] chunks = Split.allocate(lighter, source, 3, 1);
            try {
                lighter.split(new Split().source(source).into(chunks).chunks(3).dimension(1));
                for (int chunk = 0; chunk < 3; chunk++) {
                    for (int row = 0; row < 2; row++) {
                        for (int column = 0; column < 4; column++) {
                            assertEquals(row * 100.0f + chunk * 4 + column,
                                    chunks[chunk].get(row, column));
                        }
                    }
                }
            } finally {
                for (TensorRef chunk : chunks) {
                    chunk.close();
                }
            }
        }
    }
}
