package io.teknek.deliverance.model.tensorparallel.transport;

import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class BinaryTensorPayloadCodecTest {

    @Test
    public void roundTripsF32Tensor() {
        BinaryTensorPayloadCodec codec = new BinaryTensorPayloadCodec();
        try (AbstractTensor source = new FloatBufferTensor(2, 2)) {
            source.set(1.0f, 0, 0);
            source.set(2.0f, 0, 1);
            source.set(3.0f, 1, 0);
            source.set(4.0f, 1, 1);

            try (AbstractTensor decoded = codec.decode(codec.encode(source))) {
                assertEquals(2, decoded.dims());
                assertEquals(2, decoded.shape().dim(0));
                assertEquals(2, decoded.shape().dim(1));
                assertEquals(1.0f, decoded.get(0, 0));
                assertEquals(2.0f, decoded.get(0, 1));
                assertEquals(3.0f, decoded.get(1, 0));
                assertEquals(4.0f, decoded.get(1, 1));
            }
        }
    }

    @Test
    public void rejectsInvalidMagic() {
        BinaryTensorPayloadCodec codec = new BinaryTensorPayloadCodec();

        assertThrows(IllegalArgumentException.class, () -> codec.decode(new byte[]{0, 0, 0, 0}));
    }
}
