package io.teknek.deliverance.tensor;

import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TensorLocalityTest {

    @Test
    void tensorCanCarryAdvisoryLocalityMetadata() {
        try (AbstractTensor tensor = new FloatBufferTensor(2, 32)) {
            TensorLocality locality = new TensorLocality(
                    tensor.getMemorySegment().address(),
                    tensor.getMemorySegment().byteSize(),
                    1,
                    List.of(4, 5),
                    1234L,
                    "fake-test"
            );

            tensor.setLocality(locality);

            assertTrue(tensor.locality().isPresent());
            assertEquals(1, tensor.locality().orElseThrow().numaNode());
            assertEquals(List.of(4, 5), tensor.locality().orElseThrow().preferredCpus());
            assertEquals("fake-test", tensor.locality().orElseThrow().source());
        }
    }
}
