package io.teknek.deliverance.generator;

import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class VariableMLPBlockTest {

    @Test
    public void activationSparsityAppliesReluAfterMeanStdCutoff() {
        try (FloatBufferTensor gate = new FloatBufferTensor(1, 4)) {
            gate.set(1.0f, 0, 0);
            gate.set(2.0f, 0, 1);
            gate.set(3.0f, 0, 2);
            gate.set(4.0f, 0, 3);

            // mean=2.5, population std=sqrt(1.25). With multiplier 0, cutoff=mean.
            VariableMLPBlock.applyActivationSparsity(gate, 1, 4, 0.0f);

            assertEquals(0.0f, gate.get(0, 0));
            assertEquals(0.0f, gate.get(0, 1));
            assertEquals(0.5f, gate.get(0, 2));
            assertEquals(1.5f, gate.get(0, 3));
        }
    }

    @Test
    public void activationSparsityAppliesPerBatchRow() {
        try (FloatBufferTensor gate = new FloatBufferTensor(2, 4)) {
            for (int col = 0; col < 4; col++) {
                gate.set(col + 1, 0, col);
                gate.set((col + 1) * 10, 1, col);
            }

            VariableMLPBlock.applyActivationSparsity(gate, 2, 4, 0.0f);

            assertEquals(0.0f, gate.get(0, 0));
            assertEquals(0.0f, gate.get(0, 1));
            assertEquals(0.5f, gate.get(0, 2));
            assertEquals(1.5f, gate.get(0, 3));
            assertEquals(0.0f, gate.get(1, 0));
            assertEquals(0.0f, gate.get(1, 1));
            assertEquals(5.0f, gate.get(1, 2));
            assertEquals(15.0f, gate.get(1, 3));
        }
    }
}
